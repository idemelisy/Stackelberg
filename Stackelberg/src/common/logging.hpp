/// @file logging.hpp
/// @brief Lightweight compile-time logging helpers used across the solver
///
/// `LOG_INFO("message")`  – always on at level ≥ 1
/// `LOG_DEBUG("message")` – enabled only when `LOG_LEVEL >= 2`
///
/// Change the global verbosity by compiling with `-DLOG_LEVEL=n`.
#pragma once

#include <iostream>

#ifndef LOG_LEVEL
#define LOG_LEVEL 1  // 0 = silent, 1 = info, 2 = debug
#endif

#define LOG_INFO(MSG)  do { if (LOG_LEVEL >= 1) { std::cout << "[INFO]  " << MSG << std::endl; } } while (0)
#define LOG_DEBUG(MSG) do { if (LOG_LEVEL >= 2) { std::cout << "[DEBUG] " << MSG << std::endl; } } while (0) 