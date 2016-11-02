#ifndef CUDA_DBE_PREFERENCES_H_
#define CUDA_DBE_PREFERENCES_H_

//MINIMIZE | MAXIMIZE
#define MINIMIZE

#ifdef MINIMIZE
#define OP <
#define funOP min
#else
#define OP >
#define funOP max
#endif

class Preferences {
public:

  // Print and Report preferences
  static constexpr bool verbose = false;
  static constexpr bool verboseDevInit = false;
  static constexpr bool silent = true;
  static constexpr bool csvFormat = true;
  
  // PseudoTree Construction (Default parameters - 
  // can be superseeded by input)
  static constexpr int default_ptRoot      = 0;
  static constexpr int default_ptHeuristic = 2;
  static int ptRoot     ;
  static int ptHeuristic;

  // CUDA Memory preferences
  // default: when singleAgent=F, usePinned=T
  // default: when singleAgent=T, usePinned=F
  static constexpr bool usePinnedMemory = true;
  static constexpr bool singleAgent     = true;
  static constexpr float streamSizeMB = 50;
  
  // Host and Device Memory Limits (in bytes)
  static constexpr float default_maxHostMemory = 0  /*GB*/ * 1e+9;
  static constexpr float default_maxDevMemory  = 12 /*GB*/ * 1e+9; // 0 for unbounded
  static constexpr int   default_gpuDevID      = 0;

  static float maxHostMemory;
  static float maxDevMemory;
  static int   gpuDevID;
};

#endif
