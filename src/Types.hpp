//
// Created by Ferdinando Fioretto on 11/7/15.
//

#ifndef TYPES_H
#define TYPES_H

#include <limits>
#include "Preferences.hpp"

//typedef long long int power_t; // Bit encoding [1111 1111] [1111 1111] .. [] this can represent loads and generators up to optH=8
typedef unsigned short value_t;
//typedef int value_t;
typedef unsigned short util_t;
//typedef int util_t;


// Constants 
class Constants {
public:
  static constexpr value_t NaN   = std::numeric_limits<value_t>::min()+1;
  static constexpr util_t inf    = 
#ifdef MINIMIZE
    std::numeric_limits<util_t>::max();
#else
    std::numeric_limits<util_t>::min();
#endif

  static constexpr util_t unsat  = inf;
  // move this to u_math:: namespace
  static constexpr bool isFinite(util_t c) { return c != inf && c != -inf; } 
  static constexpr bool isSat(util_t c) { return c != unsat; }

  static constexpr util_t worstvalue = inf;
  // static constexpr util_t bestvalue  = inf;

  static util_t aggregate(util_t a, util_t b) {
    return (a != unsat && b != unsat)? a + b : unsat; 
  }

private:
  Constants(){ }
};

#endif //TYPES_H
