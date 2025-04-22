/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "State_JPLQuatLocal.h"

#include "utils/quat_ops.h"

using namespace ov_init;

bool State_JPLQuatLocal::Plus(const double *x, const double *delta, double *x_plus_delta) const {

  // Apply the standard JPL update: q <-- [d_th/2; 1] (x) q
  Eigen::Map<const Eigen::Vector4d> q(x);

  // Get delta into eigen 
  Eigen::Map<const Eigen::Vector3d> d_th(delta);
  Eigen::Matrix<double, 4, 1> d_q;
  double theta = d_th.norm();
  if (theta < 1e-8) {
    d_q << .5 * d_th, 1.0;
  } else {
    d_q.block(0, 0, 3, 1) = (d_th / theta) * std::sin(theta / 2);
    d_q(3, 0) = std::cos(theta / 2);
  }
  d_q = ov_core::quatnorm(d_q);

  // Do the update
  Eigen::Map<Eigen::Vector4d> q_plus(x_plus_delta);
  q_plus = ov_core::quat_multiply(d_q, q);
  return true;
}

bool State_JPLQuatLocal::PlusJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
  j.topRows<3>().setIdentity();
  j.bottomRows<1>().setZero();
  return true;
}

bool State_JPLQuatLocal::Minus(const double *y, const double *x, double *y_minus_x) const {

  // Apply the standard JPL
  Eigen::Map<const Eigen::Vector4d> q_y(y);
  Eigen::Map<const Eigen::Vector4d> q_x(x);

  // Do the update
  Eigen::Map<Eigen::Vector3d> q_minus(y_minus_x);
  Eigen::Matrix<double, 4, 1> q_minus_quat = ov_core::quat_multiply(q_y, ov_core::Inv(q_x));
  // JPL -> 李代数
  double qx = q_minus_quat(0), qy = q_minus_quat(1), qz = q_minus_quat(2), qw = q_minus_quat(3);
  // 计算整体旋转角 θ = 2 * acos(qw)
  double theta = 2.0 * std::acos(qw);
  // 计算 sin(θ/2) 并判断退化
  double sin_half = std::sqrt(1.0 - qw * qw);
  Eigen::Vector3d axis;
  if (sin_half < 1e-6) { 
      axis = Eigen::Vector3d::UnitX();  // 默认返回x轴
  } else {
      axis = Eigen::Vector3d(qx, qy, qz) / sin_half;
  }
  q_minus = theta * axis;
  return true;
}

bool State_JPLQuatLocal::MinusJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
  j.topRows<3>().setIdentity();
  j.bottomRows<1>().setZero();
  return true;
}
