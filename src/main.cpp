/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>
#include <memory>
#include <cmath>

#include "Assert.hpp"
#include "InputParams.hpp"
#include "Problem.hpp"
#include "Preferences.hpp"
#include "BE/BESolver.hpp"
#include "Utils/Timer.hpp"
#include "Utils/CpuAllocator.hpp"
#include "Utils/GpuAllocator.cuh"
#include "Utils/GpuInfo.cuh"
#include "Utils/GpuTimer.cuh"
#include "Utils/CpuInfo.hpp"


int main(int argc, char **argv) {
  Timer<> initTimer;
  Gpu::Timer::initialize();
  initTimer.start();
  
  // Parse Inputs and Initializes the GPU if necessary 
  InputParams::parse(argc, argv);
  Problem::parse(InputParams::getFileName(),
                 InputParams::getFormat(),
                 InputParams::getAgentType());

  /// Pseudo-Tree selection
  Problem::makePseudoTreeOrder(); 

  Cpu::Info::initialize();
  if (InputParams::getAgentType() == InputParams::gpuBE || 
      InputParams::getAgentType() == InputParams::gpuMiniBE) {
    Gpu::Info::initialize();
  }

  Assert::check(Problem::checkMemory(),
                "Problem cannot be solved due to memory requirement", "OOM");
  
  initTimer.pause();
  
  if (!Preferences::silent) {
    std::cout << "Solving...\n";
  }

  Timer<> wallclock;
  wallclock.start();    
  auto solver = std::make_shared<BESolver>();
  Gpu::Timer::reset();
  solver->solve();
  wallclock.pause();

  if (!Preferences::silent) {
    std::cout << solver->to_string() << '\n';
    std::cout << "Init Time : "   << initTimer.getElapsed() << " ms " << std::endl;      
    std::cout << "wc Time: " << wallclock.getElapsed() << " ms " << std::endl;
    std::cout << "sm Time: " << solver->getSimulatedTime() << " ms" << std::endl;
	std::cout << "gpu mem transfer Time: " << Gpu::Timer::getTimeMs() << " ms" << std::endl;
    std::cout << "Util   : " << solver->getProblemUtil() << std::endl;
    // std::cout << "Problem Cost: " << Problem::computeUtil() << std::endl;
  }
  else if (Preferences::csvFormat) {
	std::cout << initTimer.getElapsed() << "\t";
    std::cout << wallclock.getElapsed() << "\t";
    std::cout << solver->getSimulatedTime() << "\t";
	std::cout << Gpu::Timer::getTimeMs() << " \t";
    std::cout << solver->getProblemUtil() << std::endl;
	
	// size_t maxProjTime = 0, maxAggrTime = 0, sumProjTime = 0, sumAggrTime = 0;
	// double avgProjTime = 0, avgAggrTime = 0;
	//
	// for (size_t t : solver->getUtilPhaseProjTime()) {
	//           maxProjTime = std::max(maxProjTime, t);
	// 	sumProjTime += t;
	// }
	// avgProjTime = sumProjTime / (double)Problem::getNbAgents();
	//
	// for (size_t t : solver->getUtilPhaseAggrTime()) {
	//           maxAggrTime = std::max(maxAggrTime, t);
	// 	sumAggrTime += t;
	// }
	// avgAggrTime = sumAggrTime / (double)Problem::getNbAgents();
	//
	// std::cout << maxAggrTime << "\t" << sumAggrTime << "\t" << avgAggrTime << "\t"
	// 	 << maxProjTime << "\t" << sumProjTime << "\t" << avgProjTime << "\n";
  }

  Gpu::Allocator::freeAll();
  Cpu::Allocator::freeAll();
  return 0;
}
