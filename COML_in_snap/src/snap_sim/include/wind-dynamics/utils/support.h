#pragma once

#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <string>
#include <random>
#include <experimental/filesystem>
#include <boost/algorithm/string.hpp>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
#include <vector>

#define UTILS_PI 3.141592653589793

namespace utils {

typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;

typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;
typedef Eigen::Matrix<double, 8, 8> Matrix8d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

//template<typename Derived>
//constexpr bool is_eigen_type_f(const Eigen::EigenBase<Derived> *) {
//    return true;
//}
//constexpr bool is_eigen_type_f(const double *) {
//    return false;
//}
//constexpr bool is_eigen_type_f(const void *) {
//    return false;
//}

//template<typename T>
//constexpr bool is_eigen_type = is_eigen_type_f(reinterpret_cast<T *>(NULL));

template<typename Derived>
inline void genericSetZero(Eigen::MatrixBase<Derived> &mat)
{
    mat.setZero();
}
inline void genericSetZero(double &val)
{
    val = 0.0;
}

inline void genericSetZero(float &val)
{
    val = 0.0;
}
template<typename Derived>
inline Eigen::MatrixBase<Derived> genericElementwiseMultiply(const Eigen::MatrixBase<Derived> &A,
                                                      const Eigen::MatrixBase<Derived> &B)
{
    return A.cwiseProduct(B);
}
inline double genericElementwiseMultiply(const double &a, const double &b)
{
    return a * b;
}
inline float genericElementwiseMultiply(const float &a, const float &b)
{
    return a * b;
}
template<typename Derived>
inline Eigen::MatrixBase<Derived> genericElementwiseDivide(const Eigen::MatrixBase<Derived> &A,
                                                    const Eigen::MatrixBase<Derived> &B)
{
  return A.cwiseQuotient(B);
}
inline double genericElementwiseDivide(const double &a, const double &b)
{
    return a / b;
}
inline float genericElementwiseDivide(const float &a, const float &b)
{
    return a / b;
}
template<typename Derived>
inline Eigen::MatrixBase<Derived> genericSat(const Eigen::MatrixBase<Derived> &unsat,
                                      const Eigen::MatrixBase<Derived> &max)
{
    Eigen::MatrixBase<Derived> sat;
    for (int i = 0; i < unsat.rows(); i++)
    {
        for (int j = 0; j < unsat.cols(); j++)
        {
            sat(i, j) = (unsat(i, j) > max(i, j)) ? max(i, j) :
                        (unsat(i, j) < -1.0 * max(i, j)) ? -1.0 * max(i, j) :
                        unsat(i, j);
        }
    }
    return sat;
}
inline double genericSat(const double &unsat, const double &max)
{
    return (unsat > max) ? max : (unsat < -1.0 * max) ? -1.0 * max : unsat;
}
inline float genericSat(const float &unsat, const float &max)
{
    return (unsat > max) ? max : (unsat < -1.0 * max) ? -1.0 * max : unsat;
}
template<typename Derived>
inline double genericNorm(const Eigen::MatrixBase<Derived> &mat)
{
    return mat.norm();
}
inline double genericNorm(const double &val)
{
    return fabs(val);
}
inline float genericNorm(const float &val)
{
    return fabs(val);
}

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

template <typename T>
bool get_yaml_node(const std::string key, const std::string filename, T& val, bool print_error = true)
{
  // Try to load the YAML file
  YAML::Node node;
  try
  {
    node = YAML::LoadFile(filename);
  }
  catch (...)
  {
    std::cout << "Failed to Read yaml file " << filename << std::endl;
  }

  // Throw error if unable to load a parameter
  if (node[key])
  {
    val = node[key].as<T>();
    return true;
  }
  else
  {
    if (print_error)
    {
      throw std::runtime_error("Unable to load " + key + " from " + filename);
    }
    return false;
  }
}
template <typename Derived1>
bool get_yaml_eigen(const std::string key, const std::string filename, Eigen::MatrixBase<Derived1>& val, bool print_error=true)
{
  YAML::Node node = YAML::LoadFile(filename);
  std::vector<double> vec;
  if (node[key])
  {
    vec = node[key].as<std::vector<double>>();
    if (vec.size() == (val.rows() * val.cols()))
    {
      int k = 0;
      for (int i = 0; i < val.rows(); i++)
      {
        for (int j = 0; j < val.cols(); j++)
        {
          val(i,j) = vec[k++];
        }
      }
      return true;
    }
    else
    {
      throw std::runtime_error("Eigen Matrix Size does not match parameter size for " + key + " in " + filename +
                               ". Requested " + std::to_string(Derived1::RowsAtCompileTime) + "x" + std::to_string(Derived1::ColsAtCompileTime) +
                               ", Found " + std::to_string(vec.size()));
      return false;
    }
  }
  else if (print_error)
  {
    throw std::runtime_error("Unable to load " + key + " from " + filename);
  }
  return false;
}

template <typename Derived>
bool get_yaml_diag(const std::string key, const std::string filename, Eigen::MatrixBase<Derived>& val, bool print_error=true)
{
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1> diag;
  if (get_yaml_eigen(key, filename, diag, print_error))
  {
    val = diag.asDiagonal();
    return true;
  }
  return false;
}

template <typename T>
bool get_yaml_priority(const std::string key, const std::string file1, const std::string file2, T& val)
{
  if (get_yaml_node(key, file1, val, false))
  {
    return true;
  }
  else
  {
    return get_yaml_node(key, file2, val, true);
  }
}

template <typename Derived1>
bool get_yaml_priority_eigen(const std::string key, const std::string file1, const std::string file2, Eigen::MatrixBase<Derived1>& val)
{
  if (get_yaml_eigen(key, file1, val, false))
  {
    return true;
  }
  else
  {
    return get_yaml_eigen(key, file2, val, true);
  }
}

inline bool createDirIfNotExist(const std::string& dir)
{
  if(!std::experimental::filesystem::exists(dir))
    return std::experimental::filesystem::create_directory(dir);
  else
    return false;
}

inline std::vector<std::string> split(const std::string& s, const char* delimeter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimeter[0]))
   {
      tokens.push_back(token);
   }
   return tokens;
}

inline std::string baseName(const std::string& path)
{
  std::string filename = split(path, "/").back();
  return split(filename, ".")[0];
}

// LEGACY skew
inline Eigen::Matrix3d skew(const Eigen::Vector3d v)
{
  Eigen::Matrix3d mat;
  mat << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
         -v(1), v(0), 0.0;
  return mat;
}


static const Eigen::Matrix<double, 2, 3> I_2x3 = [] {
  Eigen::Matrix<double, 2, 3> tmp;
  tmp << 1.0, 0, 0,
         0, 1.0, 0;
  return tmp;
}();

static const Eigen::Matrix3d I_3x3 = [] {
  Eigen::Matrix3d tmp = Eigen::Matrix3d::Identity();
  return tmp;
}();

static const Eigen::Matrix2d I_2x2 = [] {
  Eigen::Matrix2d tmp = Eigen::Matrix2d::Identity();
  return tmp;
}();


static const Eigen::Vector3d e_x = [] {
  Eigen::Vector3d tmp;
  tmp << 1.0, 0, 0;
  return tmp;
}();

static const Eigen::Vector3d e_y = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 1.0, 0;
  return tmp;
}();

static const Eigen::Vector3d e_z = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, 1.0;
  return tmp;
}();

template <typename T>
Eigen::Matrix<T,3,3> skew(const Eigen::Matrix<T,3,1>& v)
{
  Eigen::Matrix<T,3,3> mat;
  mat << (T)0.0, -v(2), v(1),
         v(2), (T)0.0, -v(0),
         -v(1), v(0), (T)0.0;
  return mat;
}

template <typename Derived>
void setNormalRandom(Eigen::MatrixBase<Derived>& M, std::normal_distribution<double>& N, std::default_random_engine& g)
{
  for (int i = 0; i < M.rows(); i++)
  {
    for (int j = 0; j < M.cols(); j++)
    {
      M(i,j) = N(g);
    }
  }
}

template <typename T, int R, int C>
Eigen::Matrix<T, R, C> randomNormal(std::normal_distribution<T>& N, std::default_random_engine& g)
{
  Eigen::Matrix<T,R,C> out;
  for (int i = 0; i < R; i++)
  {
    for (int j = 0; j < C; j++)
    {
      out(i,j) = N(g);
    }
  }
  return out;
}

template <typename T, int R, int C>
Eigen::Matrix<T, R, C> randomUniform(std::uniform_real_distribution<T>& N, std::default_random_engine& g)
{
  Eigen::Matrix<T,R,C> out;
  for (int i = 0; i < R; i++)
  {
    for (int j = 0; j < C; j++)
    {
      out(i,j) = N(g);
    }
  }
  return out;
}

template <typename T>
int sign(T in)
{
  return (in >= 0) - (in < 0);
}

template <typename T>
inline T random(T max, T min)
{
  T f = (T)rand() / RAND_MAX;
  return min + f * (max - min);
}

} // end namespace utils
