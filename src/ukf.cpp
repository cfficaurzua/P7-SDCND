#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  /* A fast car (Tesla Model S P100D)   goes from 0 to 100 km/s in 2.5s -> ~11.1m/s^2
     A medium  car  (susuki Alto)       goes from 0 to 100 km/s in 10 s -> 2.78 m/s^2
     A slow car (Old beetle Volkswagen) goes from 0 to 100 km/s in 30s -> ~0.93 m/s^2
     suppose there are 80% of medium cars, 18% of slow cars and 2% of fast cars then
     mean = (80*2.78+18*0.93+2*11.1)/100 = 2.6134 m/s^2
     std = sqrt((80*(2.78-2.6134)^2+18*(0.93-2.6134)^2+2(11.1-2.6134)^2)/99) = 1.411 m/s^2
  */

  std_a_ = 1.411;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // this parameter was turn empirically
  std_yawdd_ = 0.52;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  //state dimension
  n_x_ = 5;

  //augmented dimension
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2*n_aug_+1);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  previous_time_;

  R_radar_ = MatrixXd(3,3);
  R_lidar_ = MatrixXd(2,2);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
if (!is_initialized_) {
  P_ << 1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    float rho = meas_package.raw_measurements_[0];
    float phi = meas_package.raw_measurements_[1];
    float rho_dot = meas_package.raw_measurements_[2];
    float px = rho * cos(phi);
    float py = rho * sin(phi);
    float v_x = rho_dot*cos(phi);
    float v_y = rho_dot*sin(phi);
    float v = sqrt(v_x*v_x+v_y*v_y);
    x_ << px, py, v, 0, 0;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    float px = meas_package.raw_measurements_[0];
    float py = meas_package.raw_measurements_[1];
    x_ << px, py, 0, 0, 0;
  }
  if (fabs(x_(0))<0.001){
    x_(0) = 0.001;
  }
  if (fabs(x_(1))<0.001){
    x_(1) = 0.001;
  }
  // weights initialization
  weights_.fill(0.5/(lambda_+n_aug_));
  weights_(0) = lambda_/(lambda_+n_aug_);

  R_radar_<< std_radr_*std_radr_,   0,   0,
             0, std_radphi_*std_radphi_, 0,
             0, 0,   std_radrd_*std_radrd_;

  R_lidar_ << std_laspx_*std_laspx_,0,
             0, std_laspy_*std_laspy_;

  previous_time_ = meas_package.timestamp_;
  is_initialized_ = true;
  return;
  }
/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
 double delta_t = (meas_package.timestamp_ - previous_time_)/1000000.0;
 previous_time_ = meas_package.timestamp_;
 Prediction(delta_t);
 if ( meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      UpdateRadar(meas_package);
 }
 if ( meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
   UpdateLidar(meas_package);
 }
}

double UKF::AngleNormalization(double angle) {
  return angle - 2*M_PI*floor((angle+M_PI)/(2*M_PI));
}

MatrixXd UKF::AugmentedSigmaPoints(MatrixXd Xsig_aug) {

  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  x_aug.fill(0.0);
  P_aug.fill(0.0);

  x_aug.head(n_x_) = x_;
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_)     = std_a_*std_a_;
  P_aug(n_x_+1,n_x_+1) = std_yawdd_*std_yawdd_;
  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_;i++)
  {
      Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

  return Xsig_aug;

}
MatrixXd UKF::SigmaPointPrediction(MatrixXd Xsig_aug, MatrixXd Xsig_pred_, double delta_t) {
  for (int i = 0; i< 2*n_aug_ + 1; i++){
       double px  = Xsig_aug(0,i);
       double py  = Xsig_aug(1,i);
       double v  = Xsig_aug(2,i);
       double psi  = Xsig_aug(3,i);
       double psi_d  = Xsig_aug(4,i);
       double nu_a  = Xsig_aug(5,i);
       double nu_psi_d  = Xsig_aug(6,i);
       double px_p, py_p, v_p, psi_p, psi_d_p;
       if (fabs(psi_d)>0.001) {
           px_p = px + (v/psi_d)*(sin(psi+psi_d*delta_t)-sin(psi));
           py_p = py + (v/psi_d)*(cos(psi)-cos(psi+psi_d*delta_t));
       }   else {
           px_p = px + v*cos(psi)*delta_t;
           py_p = py + v*sin(psi)*delta_t;
       }

       v_p = v ;
       psi_p = psi + psi_d*delta_t;
       psi_d_p = psi_d;

       //noise
       px_p += 0.5*delta_t*delta_t*cos(psi)*nu_a;
       py_p += 0.5*delta_t*delta_t*sin(psi)*nu_a;
       v_p  += delta_t*nu_a;
       psi_p +=  0.5*delta_t*delta_t*nu_psi_d;
       psi_d_p +=  nu_psi_d*delta_t;
       Xsig_pred_.col(i) << px_p, py_p, v_p, psi_p, psi_d_p;
   }
   return Xsig_pred_;

}
void UKF::PredictMeanAndCovariance(MatrixXd Xsig_pred_) {
  x_.fill(0.0);
  x_ = Xsig_pred_ * weights_;
  //predict state mean
  //predict state covariance matrix

  P_.fill(0.0);
  for (int i = 0; i< 2 *n_aug_+1;i++){
     VectorXd x_diff = (Xsig_pred_.col(i)-x_);
     x_diff(3) = AngleNormalization(x_diff(3));
     P_ = P_ + weights_(i)*x_diff*x_diff.transpose();
  }

}
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug = AugmentedSigmaPoints(Xsig_aug);
  Xsig_pred_ = SigmaPointPrediction(Xsig_aug, Xsig_pred_, delta_t);
  PredictMeanAndCovariance(Xsig_pred_);


}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  VectorXd z = meas_package.raw_measurements_;
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1 );
  Zsig.fill(0.0);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (int i = 0; i< 2*n_aug_+1;i++){
      double px = Xsig_pred_(0,i);
      double py = Xsig_pred_(1,i);
      Zsig.col(i) << px, py;
    }
      z_pred = Zsig * weights_;

  for (int i = 0; i < 2*n_aug_+1; i++) {
      VectorXd z_diff = Zsig.col(i) - z_pred;
      S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_lidar_;


  for (int i=0; i<2*n_aug_+1;i++){
      VectorXd z_diff = Zsig.col(i) - z_pred;
      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      x_diff(3) = AngleNormalization(x_diff(3));
      Tc = Tc + weights_(i)*(x_diff)*(z_diff).transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K_gain = Tc*S.inverse();

  VectorXd z_diff = z - z_pred;
  //update state mean and covariance matrix
  x_ = x_ + K_gain*(z_diff);
  P_ = P_ - K_gain*S*K_gain.transpose();

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //calculate cross correlation matrix

  //transform sigma points into measurement space

  int n_z = 3;
  VectorXd z = meas_package.raw_measurements_;
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1 );
  Zsig.fill(0.0);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  for (int i = 0; i< 2*n_aug_+1;i++){
      double px = Xsig_pred_(0,i);
      double py = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double psi = Xsig_pred_(3,i);
      double psi_dot = Xsig_pred_(4,i);

      double rho = sqrt(px*px+py*py);
      if(fabs(rho)<0.001){
        rho = 0.001;
      }
      double phi = atan2(py,px);
      double rho_dot = v*(px*cos(psi)+py*sin(psi))/rho;
      Zsig.col(i) << rho, phi, rho_dot;
  }
  z_pred = Zsig * weights_;
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);

  for (int i = 0; i < 2*n_aug_+1; i++) {

      VectorXd z_diff = Zsig.col(i) - z_pred;
      z_diff(1) = AngleNormalization(z_diff(1));
      S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S = S + R_radar_;

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i=0; i<2*n_aug_+1;i++){
      VectorXd z_diff = Zsig.col(i)-z_pred;
      VectorXd x_diff = Xsig_pred_.col(i)-x_;
      z_diff(1) = AngleNormalization(z_diff(1));
      x_diff(3) = AngleNormalization(x_diff(3));
      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K_gain = Tc*S.inverse();

  VectorXd z_diff = z - z_pred;
  z_diff(1) = AngleNormalization(z_diff(1));
  //update state mean and covariance matrix
  x_ = x_ + K_gain*z_diff;
  P_ = P_ - K_gain*S*K_gain.transpose();
}
