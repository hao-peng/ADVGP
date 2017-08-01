#include "Eigen/Dense"
#include "Eigen/Cholesky"
#include "ps/ps.h"
#include "unordered_set"
#include <fstream>
#include <map>
#include <chrono>

using namespace ps;
using namespace Eigen; 
using namespace std::chrono;

//const double PI = acos(-1.0);


// TODO: could replace this with protobuf
class Config {
public:
  int _m;  // number of inducing inputs
  int _d;  // dimension of inputs
  int _maxDelay; // maximum delay allowed for asynchronous updates
  int _maxPush;  // maximum number of pushes before wait
  int _maxEpoch; // maximum number of epoches

  // parameters for adadelta
  // see paper https://arxiv.org/pdf/1212.5701.pdf
  double _tau; // rho in the paper 
  double _eps;
  
  double _fixedJitter; // fixed jitter for numerical stability 

  // paths for inputs and outputs
  std::string _trainFilePath;
  std::string _testFilePath;
  std::string _paramFilePath;
  std::string _outputFilePath;

  // offsets of parameters
  size_t _snOffset;
  size_t _ellOffset;
  size_t _sfOffset;
  size_t _bOffset;
  size_t _wMuOffset;
  size_t _wCholSOffset;  // only store the upper triangular parts for wCholS
  size_t _paramSize;
  size_t _lossOffset;
  size_t _epochOffset;
  size_t _timeOffset;
  size_t _totalKeys;

  Config() : _m(0), _d(0) {}
  // Read the configuration from a file located at configFilePath.
  // This is a simple function which assumes that the inputs are stictly
  // ordered. Lines begin with # are comments.
  void Load(std::string configFilePath) {
    std::ifstream ifs(configFilePath.c_str(), std::ifstream::in);
    int line = 0;
    char buf[1024];
    while (ifs.good()) {
      ifs.getline(buf, 1024);
      if (buf[0] == '#') continue;
      switch (line) {
        case 0: _m = std::stoi(buf); break; 
        case 1: _d = std::stoi(buf); break;
        case 2: _tau = std::stod(buf); break;
        case 3: _eps = std::stod(buf); break;
        case 4: _fixedJitter = std::stod(buf); break;
        case 5: _maxPush = std::stoi(buf); break;
        case 6: _maxEpoch = std::stoi(buf); break;
        case 7: _maxDelay = std::stoi(buf); break;
        case 8: _trainFilePath = buf; break;
        case 9: _testFilePath = buf; break;
        case 10: _paramFilePath = buf; break;
        case 11: _outputFilePath = buf; break;
      }
      line++;
    }
    ifs.close();

    // compute offsets
    _snOffset     = 0;
    _ellOffset    = _snOffset     + 1;
    _sfOffset     = _ellOffset    + _d;
    _bOffset      = _sfOffset     + 1;
    _wMuOffset    = _bOffset      + _d * _m;
    _wCholSOffset = _wMuOffset    + _m; 
    // only store the upper triangular parts for wCholS
    _paramSize    = _wCholSOffset + _m * (_m + 1) / 2;
    _lossOffset   = _paramSize;
    _epochOffset  = _paramSize    + 1;
    _timeOffset   = _epochOffset  + 1;
    _totalKeys    = _timeOffset   + 1;
    
    // check config 
    CHECK_GT(_m, 0);
    CHECK_GT(_d, 0);
    CHECK_GT(_maxPush, 0);
    CHECK_GT(_tau, 0.0);
    CHECK_GT(line, 11);
  }

  // This converts the configureation to a string.
  std::string toString() {
    return "# M\n" + std::to_string(_m)
       + "\n# D\n" + std::to_string(_d)
       + "\n# tau\n" + std::to_string(_tau)
       + "\n# eps\n" + std::to_string(_eps)
       + "\n# fixed jitter\n" + std::to_string(_fixedJitter)
       + "\n# max push\n" + std::to_string(_maxPush)
       + "\n# max epoch\n" + std::to_string(_maxEpoch)
       + "\n# max delay\n" + std::to_string(_maxDelay)
       + "\n# train file\n" + _trainFilePath
       + "\n# test file\n" + _testFilePath
       + "\n# output file\n" + _outputFilePath
       + "\n# parameter file\n" + _paramFilePath + "\n";
  }
};

Config config;

// use revertible mapping to distribute the keys among servers
inline Key encryptKey(Key key) {
  return kMaxKey / config._totalKeys * key;
}
inline Key decryptKey(Key key) {
  return key / (kMaxKey / config._totalKeys);
}

// Handler for the server (it is used when a push/pull request comes)
struct MyKVServerHandle {
  MyKVServerHandle() {
    _numTrainers = NumWorkers() - 1; 
    _grads.resize(_numTrainers);
    _losses.resize(_numTrainers);
    _loss = 0.0;
    _epochs.resize(_numTrainers);
    _epochCounts[0] = _numTrainers;
    _curEpoch = 0;

    LoadParameters();

    // set the indexes for diagonals of wCholS for fast look-up
    // see Eq. (18 - 20)
    int ind = config._wCholSOffset;
    for (int i = 0; i < config._m; i++) {
      _wSDiagInd.insert(ind);
      ind += config._m - i;
    }
    
    _startTime = high_resolution_clock::now();

    _pushTime = 0;
    _pushCount = 0;
    _processTime = 0;
    _processCount = 0;
  }
  
  // param = [ln(sn) ln(ell) ln(sf) B    m    chol(S)]
  // size     1      d       1      md   m    m(m+1)/2
  // B and chol(S) are stored in row major
  void LoadParameters() {
    std::ifstream ifs(config._paramFilePath, std::ifstream::in);
    double data;
    size_t ind = 0;
    while (ifs >> data) {
      _params[ind++] = data;
    }
    ifs.close();
    CHECK_EQ(ind, config._paramSize);
  }

  // process a push request
  // If the incoming updates are within the allowed delay, updates the
  // parameters and process all unhanedled pull requests
  inline void ProcessPush(const KVMeta &req_meta, 
                          const KVPairs<double>& req_data, 
                          KVServer<double> *server, 
                          KVPairs<double> &res) {
    size_t n = req_data.keys.size();
    CHECK_EQ(n, req_data.vals.size());
    int rank = Postoffice::Get()->IDtoRank(req_meta.sender);
    // only update if the new epoch is larger
    if (req_meta.cmd < _epochs[rank]) return;

    if (--_epochCounts[_epochs[rank]] == 0) {
      _epochCounts.erase(_epochs[rank]);
    }
    _epochs[rank] = _curEpoch + 1;
    _epochCounts[_epochs[rank]]++;

    for (size_t i = 0; i < n; ++i) {
      Key key = decryptKey(req_data.keys[i]);
      if (key == config._lossOffset) { // loss
        _losses[rank] = req_data.vals[i];
      } else if (key == config._epochOffset) { // epoch
        // do nothing
      } else {
        _grads[rank][key] = req_data.vals[i];
      }
    }

    // compute the min epoch among all workers
    int minEpoch = _epochCounts.begin()->first;

    // only update parameter if all workers are within maxDelay
    if (_curEpoch + 1 - minEpoch > config._maxDelay) return;
    _curEpoch++;

    //LL << "server update " << _curEpoch;

    for (size_t i = 0; i < n; ++i) {
      Key key = decryptKey(req_data.keys[i]);
      if (key == config._lossOffset) {
        _loss = 0.0;
        for (size_t j = 0; j < _numTrainers; ++j) {
          _loss += _losses[j];
        }
      } else if (key == config._epochOffset) {
        // do nothing
      } else if (key == config._timeOffset) { 
        // do nothing
      } else {
        double gradSum = 0.0;
        for (size_t j = 0; j < _numTrainers; ++j) {
          gradSum += _grads[j][key];
        }

        // AdaDelta algorithm
        _accumGrad[key] = _accumGrad[key] * config._tau + 
                          gradSum * gradSum * (1 - config._tau);
        double stepSize = sqrt(_accumUpd[key] + config._eps) /
                          sqrt(_accumGrad[key] + config._eps);
        double update = - stepSize * gradSum;
        double gradDesUpd = _params[key] + update;
        _accumUpd[key] = _accumUpd[key] * config._tau +
                         update * update * (1 - config._tau);

        if (i < config._wMuOffset) {
          // asychronous gradient updates for non-variational parameters 
          _params[key] = gradDesUpd;
        } else {
          if (_wSDiagInd.find(key) == _wSDiagInd.end()) {
            // proximal update for regular L2 loss
            // Eq. (18) and (19) in ADVGP paper
            _params[key] = gradDesUpd / (1.0 + stepSize);
          } else {
            // different update for diagonals of wCholS
            // Eq. (20) in ADVGP paper
            _params[key] = (gradDesUpd + sqrt(gradDesUpd * gradDesUpd 
                      + 4.0 * (1.0 + stepSize) * stepSize)) 
                      * 0.5 / ( 1.0 + stepSize);
          }
        }
      }
    }  // end of loop over all keys
    
    // process unhandled pull requests
    std::vector<int> toDel;
    for (std::unordered_map<int, KVMeta>::iterator ii = _reqMetas.begin();
        ii != _reqMetas.end(); ii++) {
      if (_curEpoch > ii->second.cmd) {
        KVPairs<double> res;
        ProcessPull(ii->second, _reqData[ii->first], server, res);
        server->Response(ii->second, res);
        toDel.push_back(ii->first);
      }
    }
    for (size_t i = 0; i < toDel.size(); i++) {
      _reqMetas.erase(toDel[i]);
      _reqData.erase(toDel[i]);
    }
  }  // ProcessPush


  // process a pull request
  // send all necessary information to the worker
  inline void ProcessPull(const KVMeta &req_meta, 
                          const KVPairs<double>& req_data, 
                          KVServer<double> *server, 
                          KVPairs<double> &res) {
    size_t n = req_data.keys.size();

    res.keys = req_data.keys;
    res.vals.resize(n);
    for (size_t i = 0; i < n; ++i) {
      Key key = decryptKey(req_data.keys[i]);
      if (key == config._lossOffset) {
        res.vals[i] = _loss;
      } else if (key == config._epochOffset) {
        res.vals[i] = _curEpoch;
      } else if (key == config._timeOffset) {
        high_resolution_clock::time_point t = high_resolution_clock::now();
        duration<double> dur = t - _startTime;
        res.vals[i] = dur.count();
      } else {
        res.vals[i] = _params[key];
      }
    }
  }  // ProcessPull


  // The function for all requests (pull/push)
  // The req_meta.push is boolean that indicate whether it is a pull or a push
  // req_meta.cmd stores the epoch from the worker
  void operator()(const KVMeta& req_meta, 
                  const KVPairs<double>& req_data, 
                  KVServer<double>* server) {
    KVPairs<double> res;

    if (req_meta.push) {
      ProcessPush(req_meta, req_data, server, res);
      server->Response(req_meta, res);
    } else {
      // for Pull request, send immediately if _curEpoch is larger
      // otherwise store the pull request utill the next push update
      if (req_meta.cmd < _curEpoch) {
        ProcessPull(req_meta, req_data, server, res);
        server->Response(req_meta, res);
      } else {
        int rank = Postoffice::Get()->IDtoRank(req_meta.sender);
        _reqMetas[rank]  = req_meta;
        _reqData[rank] = req_data;
      }
    }
  }

  // gradients for each parameter and for each worker
  std::vector< std::unordered_map<Key, double> > _grads;
  // parameters (hyperperatmers and parameters of ADVGP model)
  std::unordered_map<Key, double>                _params;
  // a helper index for fast look-up when updating gradients
  std::unordered_set<int>                        _wSDiagInd;
  // total loss (variational lower bound)
  double                                         _loss;
  // losses from each worker
  std::vector<double>                            _losses;
  // current epoch for each worker
  std::vector<int>                               _epochs;
  // stores a sorted count of epoch number (for fast look-up)
  std::map<int, int>                             _epochCounts;
  // current epoch on server (max epoch)
  int                                            _curEpoch;
  // start time of this server
  high_resolution_clock::time_point              _startTime;
  // number of workers (monitors/testers excluded)
  size_t                                         _numTrainers;
  // accumultated gradients and updates used for AdaDelta algorithm
  std::unordered_map<Key, double>                _accumGrad;
  std::unordered_map<Key, double>                _accumUpd;
  // request data for unhandled pull requests
  std::unordered_map<int, KVMeta>                _reqMetas;
  std::unordered_map<int, KVPairs<double> >      _reqData;
  
  double _pushTime;
  int _pushCount;
  double _processTime;
  int _processCount;
};

// a super class for trainer and monitor/tester
class MyWorker {
protected:
  KVWorker<double> *_kvPtr;
  int _rank;

  double   _sn2;
  VectorXd _eta;    // d: inv(ell)^2
  double   _sf2;    // 1: a0 in ADVGP paper
  MatrixXd _b;      // d by m
  VectorXd _wMu;    // m
  MatrixXd _wCholS; // m by m upper

  double _loss;

  double _time;

  std::vector<Key>    _keys;
  std::vector<double> _params;

  int _curEpoch;

  double (*_exp) (double);

  // dataFilePath contains the data with N rows and (D+1) cols.
  // The last col is the response variable
  // xRaw is column major D by N matrix
  void LoadTextData(const std::string &dataFilePath, 
    std::vector<double> &xRaw, std::vector<double> &yRaw) {
    xRaw.resize(0);
    yRaw.resize(0);
    std::ifstream ifs(dataFilePath, std::ifstream::in);
    int ind = 0;
    double data;
    while (ifs >> data) {
      if (ind % (config._d + 1) == config._d) {
        yRaw.push_back(data);
      } else {
        xRaw.push_back(data);
      }
      ind ++;
    }
    ifs.close();

    //CHECK_GT(ind, 0);
    CHECK_EQ(ind % (config._d + 1), 0);
  }

public:
  MyWorker (int rank) : _rank(rank) {
    _kvPtr = new KVWorker<double> (0);

    _sn2     = 1.0;
    _eta     = VectorXd::Zero(config._d);
    _sf2     = 1.0;
    _b       = MatrixXd::Zero(config._m, config._d);
    _wMu     = VectorXd::Zero(config._m);
    _wCholS  = MatrixXd::Zero(config._m, config._m);

    _loss = 0.0;

    _keys.resize(config._totalKeys);
    for (size_t i = 0; i < config._totalKeys; i++) {
      _keys[i] = encryptKey(i);
    }
    _params.resize(config._totalKeys);

    _curEpoch = -1;

    _exp = &std::exp; // pointer to exp(double) function
  }

  virtual ~MyWorker() {
    delete _kvPtr;
  }

  virtual void Pull() {
    _kvPtr->Wait(_kvPtr->Pull(_keys, &_params, {}, _curEpoch));
    _curEpoch = _params[config._epochOffset];

    _sn2 = exp(2.0 * _params[config._snOffset]);
    for (int i = 0; i < config._d; i++) {
      _eta(i) = exp(-2.0 * _params[i + config._ellOffset]);
    }
    _sf2 = exp(2.0 * _params[config._sfOffset]);
    for (int i = 0; i < config._m; i++) {
      for (int j = 0; j < config._d; j++) {
        _b(i, j) = _params[i * config._d + j + config._bOffset];
      }
    }
    for (int i = 0; i < config._m; i++) {
      _wMu(i) = _params[i + config._wMuOffset];
    }
    for (int i = 0, offset = 0; i < config._m; i++) {
      for (int j = i; j < config._m; j++, offset++) {
        _wCholS(i, j) = _params[offset + config._wCholSOffset];
      }
    }
    _loss = _params[config._lossOffset];
    _time = _params[config._timeOffset];
  }

  virtual void Push() {}

  virtual void Process() {}

  virtual void Finalize() {}

  virtual inline bool IsFinished() {
    return _curEpoch >= config._maxEpoch;
  }
};

class MyTrainer : public MyWorker {
protected:
  std::vector<double> _updates;
  std::queue<int> _ts; // timestamps for pull

  double   _gradLnSn;
  VectorXd _gradLnEll;
  double   _gradLnSf;
  MatrixXd _gradB;
  VectorXd _gradWMu;
  MatrixXd _gradWCholS;

  MatrixXd _x;      // n * d
  VectorXd _y;      // n
  int _n;

  // pre-allocated matrix/vectors 
  MatrixXd _wS;
  MatrixXd _wSPlusWMuWMuT;  //wS + wMu * wMu^T
  MatrixXd _bEta;
  MatrixXd _kmm;
  MatrixXd _knm;
  MatrixXd _invCholInvKmm;  // inv(chol(inv(Kmm)))
  MatrixXd _cholInvKmm;  // chol(inv(Kmm))
  MatrixXd _phi;
  MatrixXd _phi2;  // _phi .* _phi
  MatrixXd _phiPhi;  // _phi^T * _phi
  RowVectorXd _yPhi;  // _y * _phi
  MatrixXd _psi;
  MatrixXd _x2;  // _x .* _x
  MatrixXd _b2;  // _b .* _b
  MatrixXd _e;  //_wSPlusWMuWMuT - I
  MatrixXd _cholInvKmmEKmn;
  MatrixXd _s;  //cholInvKmmEKmn * x
  MatrixXd _eKnm;
  MatrixXd _f;  //(cholInvKmm'*((cholInvKmm*kKnm').*psi)*cholInvKmm).*Kmm
  MatrixXd _ff;

public:
  MyTrainer(int r) : MyWorker(r) {
    CHECK_NE(r, NumWorkers() - 1);

    _updates.resize(config._totalKeys);

    _gradLnSn   = 0.0;
    _gradLnEll  = VectorXd::Zero(config._d);
    _gradLnSf   = 0.0;
    _gradB      = MatrixXd::Zero(config._m, config._d);
    _gradWMu    = VectorXd::Zero(config._m);
    _gradWCholS = MatrixXd::Zero(config._m, config._m);

    std::vector<double> _xRaw;
    std::vector<double> _yRaw;
    LoadTextData(config._trainFilePath+std::to_string(r)+".txt", _xRaw, _yRaw);
    _n = _yRaw.size();
    if (_n == 0) {
      LL << "Trainer " << _rank << " has no data.";
      return;
    }

    _x = Map<MatrixXd> (_xRaw.data(), config._d, _n).transpose();
    _y = Map<VectorXd> (_yRaw.data(), _n);

    _wS             = MatrixXd::Zero(config._m, config._m);
    _wSPlusWMuWMuT  = MatrixXd::Zero(config._m, config._m);
    _bEta           = MatrixXd::Zero(config._m, config._d);
    _kmm            = MatrixXd::Zero(config._m, config._m);
    _knm            = MatrixXd::Zero(_n, config._m);
    _invCholInvKmm  = MatrixXd::Zero(config._m, config._m);
    _cholInvKmm     = MatrixXd::Zero(config._m, config._m);
    _phi            = MatrixXd::Zero(_n, config._m);
    _phi2           = MatrixXd::Zero(_n, config._m);
    _phiPhi         = MatrixXd::Zero(config._m, config._m);
    _yPhi           = RowVectorXd::Zero(config._m);
    _psi            = MatrixXd::Zero(config._m, config._m);
    _psi.triangularView<StrictlyLower>().fill(1.0);
    _psi.diagonal().fill(0.5);
    _x2.noalias()   = _x.cwiseProduct(_x);
    _b2             = MatrixXd::Zero(config._m, config._d);
    _e              = MatrixXd::Zero(config._m, config._m);
    //_e              = MatrixXd::Zero(config._m, _n);
    _cholInvKmmEKmn = MatrixXd::Zero(config._m, _n);
    _s              = MatrixXd::Zero(config._m, config._d);
    _eKnm           = MatrixXd::Zero(config._m, config._m);
    _f              = MatrixXd::Zero(config._m, config._m);
    _ff             = MatrixXd::Zero(config._m, config._m);
    
    //_totalStart = high_resolution_clock::now(); 
  }

  void Push() {
    _updates[config._snOffset] = _gradLnSn;
    for (int i = 0; i < config._d; i++) {
      _updates[i + config._ellOffset] = _gradLnEll(i);
    }
    _updates[config._sfOffset ] = _gradLnSf;
    for (int i = 0; i < config._m; i++) {
      for (int j = 0; j < config._d; j++) {
        _updates[i * config._d + j + config._bOffset] = _gradB(i, j);
      }
    }
    for (int i = 0; i < config._m; i++) {
      _updates[i + config._wMuOffset] = _gradWMu(i);
    }

    for (int i = 0, offset = 0; i < config._m; i++) {
      for (int j = i; j < config._m; j++, offset++) {
        _updates[offset + config._wCholSOffset] = _gradWCholS(i, j);
      }
    }
    _updates[config._lossOffset]  = _loss;
    _updates[config._epochOffset] = _curEpoch;
    _updates[config._timeOffset]  = _time;

    _ts.push(_kvPtr->Push(_keys, _updates, {}, _curEpoch));
    while (_ts.size() >= (size_t) config._maxPush) {
      _kvPtr->Wait(_ts.front());
      _ts.pop();
    }
  }

  void Process() {
    if (_n == 0) return;

    _wS.noalias() = _wCholS.transpose() * _wCholS;
    _wSPlusWMuWMuT = _wS;
    _wSPlusWMuWMuT += _wMu * _wMu.transpose();

    _b2.noalias() = _b.cwiseProduct(_b);
    _bEta.noalias() = _b * _eta.asDiagonal();  

    _kmm.noalias()          = _bEta * _b.transpose();
    _kmm.colwise()          -= 0.5 * (_b2 * _eta);
    _kmm.rowwise()          -= 0.5 * (_b2 * _eta).transpose();
    _kmm.noalias()          =  _sf2 * _kmm.unaryExpr(_exp);
    _kmm.diagonal().array() += config._fixedJitter;

    _knm.noalias() = _x * _bEta.transpose();
    _knm.colwise() -= 0.5 * (_x2 * _eta);
    _knm.rowwise() -= 0.5 * (_b2 * _eta).transpose();
    _knm.noalias() =  _sf2 * _knm.unaryExpr(_exp);

    _invCholInvKmm = _kmm.reverse().llt().matrixU();
    std::reverse(_invCholInvKmm.data(), _invCholInvKmm.data()
                 + _invCholInvKmm.size());
    _cholInvKmm = MatrixXd::Identity(config._m, config._m);
    _invCholInvKmm.transpose().triangularView<Upper>().solveInPlace(
                                                              _cholInvKmm);

    // basis function. Eq. (11) in ADVGP paper 
    _phi.noalias()    = _knm * _cholInvKmm.transpose();
    _phi2.noalias()   = _phi.cwiseProduct(_phi);
    _phiPhi.noalias() = _phi.transpose() * _phi;
    _yPhi.noalias()   = _y.transpose() * _phi;

    // _e = _wS + wMu *wMu^T - I.
    // Regroup the term with phi_i * (...) * phi_i^T in Eq. (1) (4) (5)
    // in ADVGP paper. _e is the (...) part
    _e.noalias() = _wSPlusWMuWMuT;
    _e.diagonal().array() -= 1.0;
    
    _cholInvKmmEKmn.noalias() = _cholInvKmm.transpose() * _e * _phi.transpose()
      -_cholInvKmm.transpose() * _wMu * _y.transpose();
    
    _cholInvKmmEKmn.noalias() = _cholInvKmmEKmn.cwiseProduct(_knm.transpose());

    _s.noalias() = _cholInvKmmEKmn * _x;

    _eKnm.noalias() =  _e * (_phi.transpose() * _knm)
      - _wMu * (_y.transpose() * _knm);

    // Eq. (8) in ADVGP appendix
    _f.noalias()    = (_cholInvKmm.transpose() * (
                  _cholInvKmm * _eKnm.transpose()).cwiseProduct(_psi) 
                  * _cholInvKmm).cwiseProduct(_kmm);
    _ff.noalias()   = _f + _f.transpose();

    // g : local loss. Eq. (1) in ADVGP Appendix
    _loss = 0.5 * (_n * log(2.0 * M_PI) + _n * log(_sn2) +
          (_y.dot(_y) - 2.0 * _yPhi.dot(_wMu) + 
          _phiPhi.cwiseProduct(_wSPlusWMuWMuT).sum() +
          _sf2 * _n - _phi2.sum()) / _sn2);

    // d g / d ln(sn). Eq. (4) in ADVGP Appendix
    _gradLnSn = _n - (_y.dot(_y) - 2.0 * _yPhi.dot(_wMu) +
          _phiPhi.cwiseProduct(_wSPlusWMuWMuT).sum() + 
          _sf2 * _n - _phi2.sum()) / _sn2;

    // d g / d ln(ell). Eq. (10) in ADVGP Appendix
    _gradLnEll.noalias() = (2.0 * _s.cwiseProduct(_b).colwise().sum() -
        _cholInvKmmEKmn.colwise().sum() * _x2 - 
        _cholInvKmmEKmn.rowwise().sum().transpose() * _b2 -
        _b.cwiseProduct(_ff * _b).colwise().sum() +
        _ff.colwise().sum() * _b2).transpose().cwiseProduct(_eta) / -_sn2;

    // d g / d ln(sf). Eq. (5) in ADVGP Appendix
    _gradLnSf = (-_yPhi.dot(_wMu) +
          _phiPhi.cwiseProduct(_wSPlusWMuWMuT).sum() +
          _sf2 * _n - _phi2.sum()) / _sn2;

    // d g / d b. Eq(6) in ADVGP Appendix
    _gradB.noalias() = (_s * _eta.asDiagonal() - _cholInvKmmEKmn
          .rowwise().sum().replicate(1, config._d).cwiseProduct(_bEta) -
          _ff * _bEta +
          (_ff.rowwise().sum() * _eta.transpose()).cwiseProduct(_b)) / _sn2;

    // d g / d wMu
    _gradWMu.noalias() = (-_yPhi.transpose() + _phiPhi * _wMu) / _sn2;

    // d g / d wCholS
    _gradWCholS.noalias() = _wCholS * _phiPhi / _sn2;
  }

  void Finalize() {
    while (_ts.size() > 0) {
      _kvPtr->Wait(_ts.front());
      _ts.pop();
    }
  }

};

// a structure for outputs
// TODO: can be replaced with protobuf
struct Record {
public:
  double _g;
  double _h;
  double _rmse;
  double _nmse;
  double _mnlp;
  double _time;

  Record() : _g(std::numeric_limits<double>::min()), 
             _h(std::numeric_limits<double>::min()),
             _rmse(0.0), _nmse(0.0), _mnlp(0.0), _time(0) {}
};


// A monitor just can either monitor the progress of the trainer
// or work as a tester to test new dataset
// derivations not mentioned in the paper
// Please see matlab code pred.m for derivations
class MyMonitor : public MyWorker {
protected:
  MatrixXd _x;      // n * d
  VectorXd _y;      // n
  int _n;

  std::vector<Record> _records;
  std::ofstream res;

  MatrixXd _wS;
  MatrixXd _bEta;
  MatrixXd _kmm;
  MatrixXd _knm;
  MatrixXd _invCholInvKmm;
  MatrixXd _cholInvKmm;
  MatrixXd _phi;
  MatrixXd _phi2;
  MatrixXd _phiWS;
  MatrixXd _phiDotPhiWS;
  MatrixXd _x2;
  MatrixXd _b2;
  double   _meanY;

  VectorXd _predMu; // predictive mean
  VectorXd _predS2; // predictive variance

  VectorXd _modelErr2;
  VectorXd _meanErr2;
  VectorXd _scaledModelErr2;

public:
  MyMonitor(int r) : MyWorker(r) {
    CHECK_EQ(r, NumWorkers() - 1);
    LL << config.toString();

    std::vector<double> _xRaw;
    std::vector<double> _yRaw;
    LoadTextData(config._testFilePath, _xRaw, _yRaw);
    _n = _yRaw.size();
    if (_n == 0) {
      LL << "Monitor has no data.";
      return;
    }

    _x = Map<MatrixXd> (_xRaw.data(), config._d, _n).transpose();
    _y = Map<VectorXd> (_yRaw.data(), _n);

    _wS             = MatrixXd::Zero(config._m, config._m);
    _bEta           = MatrixXd::Zero(config._m, config._d);
    _kmm            = MatrixXd::Zero(config._m, config._m);
    _knm            = MatrixXd::Zero(_n, config._m);
    _invCholInvKmm  = MatrixXd::Zero(config._m, config._m);
    _cholInvKmm     = MatrixXd::Zero(config._m, config._m);
    _phi            = MatrixXd::Zero(_n, config._m);
    _phi2           = MatrixXd::Zero(_n, config._m);
    _phiWS          = MatrixXd::Zero(_n, config._m);
    _phiDotPhiWS    = MatrixXd::Zero(_n, config._m);
    _x2.noalias()   = _x.cwiseProduct(_x);
    _b2             = MatrixXd::Zero(config._m, config._d);
    _meanY = _y.sum() / _n;

    _predMu = VectorXd::Zero(_n);
    _predS2 = VectorXd::Zero(_n);

    _modelErr2       = VectorXd::Zero(_n);
    _meanErr2        = VectorXd::Zero(_n);
    _scaledModelErr2 = VectorXd::Zero(_n);

    res.open(config._outputFilePath + ".rec", std::ofstream::out);
    res << "Epoch\ttime\tf\tg\th\trmse\tnmse\tmnlp" << std::endl;
    
    //_totalStart = high_resolution_clock::now(); 
  }


  void Process() {
    _records.resize(_curEpoch + 1);

    _records[_curEpoch]._h = 
        0.5 * (-2.0 * _wCholS.diagonal().array().abs().log().sum() -
        config._m + _wMu.dot(_wMu) + _wCholS.cwiseProduct(_wCholS).sum());
    _records[_curEpoch]._g = _loss;
    _records[_curEpoch]._time = _time;

    if (_n != 0) {
      // test data
      _wS.noalias() = _wCholS.transpose() * _wCholS;

      _b2.noalias() = _b.cwiseProduct(_b);
      _bEta.noalias() = _b * _eta.asDiagonal();  

      _kmm.noalias()          = _bEta * _b.transpose();
      _kmm.colwise()          -= 0.5 * (_b2 * _eta);
      _kmm.rowwise()          -= 0.5 * (_b2 * _eta).transpose();
      _kmm.noalias()          =  _sf2 * _kmm.unaryExpr(_exp);
      _kmm.diagonal().array() += config._fixedJitter;

      _knm.noalias() = _x * _bEta.transpose();
      _knm.colwise() -= 0.5 * (_x2 * _eta);
      _knm.rowwise() -= 0.5 * (_b2 * _eta).transpose();
      _knm.noalias() =  _sf2 * _knm.unaryExpr(_exp);

      _invCholInvKmm = _kmm.reverse().llt().matrixU();
      std::reverse(_invCholInvKmm.data(), _invCholInvKmm.data() 
                                          + _invCholInvKmm.size());
      _cholInvKmm = MatrixXd::Identity(config._m, config._m);
      _invCholInvKmm.transpose().triangularView<Upper>().solveInPlace(
                                                              _cholInvKmm);

      // basis function
      _phi.noalias()    = _knm * _cholInvKmm.transpose();
      _phi2.noalias()   = _phi.cwiseProduct(_phi);

      _phiWS.noalias()       = _phi * _wS;
      _phiDotPhiWS.noalias() = _phi.cwiseProduct(_phiWS);

      // predictive distribution
      _predMu.noalias() = _phi * _wMu;
      _predS2.noalias() = -_phi2.rowwise().sum() +
                          _phiDotPhiWS.rowwise().sum();
      _predS2.array()   += _sn2 + _sf2;

      _modelErr2.noalias()       = _y - _predMu;
      _modelErr2.noalias()       = _modelErr2.cwiseProduct(_modelErr2);
      _meanErr2.noalias()        = _y;
      _meanErr2.array()          -= _meanY;
      _meanErr2.noalias()        = _meanErr2.cwiseProduct(_meanErr2);
      _scaledModelErr2.noalias() = _modelErr2.cwiseQuotient(_predS2);

      double sumModelErr2 = _modelErr2.sum();
      _records[_curEpoch]._rmse = sqrt(sumModelErr2 / _n);
      _records[_curEpoch]._nmse = sumModelErr2 / _meanErr2.sum();
      _records[_curEpoch]._mnlp = 0.5 * (_scaledModelErr2.sum() / _n +
                        _predS2.array().log().sum() / _n + log(2.0 * M_PI));
    }

    if (_curEpoch == 0) return;

    LL << "(Epoch " << _curEpoch
       << ") g: "   << _records[_curEpoch]._g 
       << " h: "    << _records[_curEpoch]._h
       << " RMSE: " << _records[_curEpoch]._rmse
       << " NMSE: " << _records[_curEpoch]._nmse 
       << " MNLP: " << _records[_curEpoch]._mnlp;

    res << _curEpoch 
      << "\t" << _records[_curEpoch]._time 
      << "\t" << _records[_curEpoch]._h + _records[_curEpoch]._g 
      << "\t" << _records[_curEpoch]._g 
      << "\t" << _records[_curEpoch]._h 
      << "\t" << _records[_curEpoch]._rmse
      << "\t" << _records[_curEpoch]._nmse 
      << "\t" << _records[_curEpoch]._mnlp << std::endl;
  }

  void Finalize() {
    res.close();

    std::ofstream ofs;
    // output params
    ofs.open(config._outputFilePath + ".param", std::ofstream::out);
    for (unsigned i = 0; i < config._paramSize; i++) {
      ofs << _params[i] << " ";
    }
    ofs.close();

    if (_n == 0) return;
    // output predictions
    ofs.open(config._outputFilePath + ".pred", std::ofstream::out);
    ofs << "PredMean\tPredVar" << std::endl;
    for (int i = 0; i < _n; i++) {
      ofs << _predMu(i) << "\t" << _predS2(i) << std::endl;
    }
    ofs.close();
  }
};


void StartServer() {
  if (!IsServer()) return;
  auto server = new KVServer<double>(0);
  server->set_request_handle(MyKVServerHandle());
  RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!IsWorker()) return;
  int rank = MyRank();
  MyWorker *wk = NULL;
  
  if (rank == NumWorkers() - 1) {
    wk = new MyMonitor(rank);
  } else {
    wk = new MyTrainer(rank);
  }

  LL << "worker " << rank << " ready!";

  while (!wk->IsFinished()) {
    wk->Pull();
    wk->Process();
    wk->Push();
  }
  wk->Finalize();

  delete wk;
}

int main(int argc, char *argv[]) {
  // load configure file
  if (argc != 2) {
    LOG(ERROR) << "Usage: " << argv[0] << " [config]" << std::endl;
    return 1;
  }
  config.Load(argv[1]);

  // setup server nodes
  StartServer();
  // start system
  Start();
  // run worker nodes
  RunWorker();
  // stop system
  Finalize();

  return 0;
}
