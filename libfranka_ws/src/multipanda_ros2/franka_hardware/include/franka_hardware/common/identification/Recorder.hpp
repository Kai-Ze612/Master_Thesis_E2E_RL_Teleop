/** ------------------------- Revision Code History -------------------
*** Programming Language: C++
*** Description: Data Recorde
*** Released Date: Feb. 2021
*** Hamid Sadeghian
*** h.sadeghian@eng.ui.ac.ir
----------------------------------------------------------------------- */

#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace panda_identification {
  class Recorder {
  public:
    Recorder(double t_rec, double sampleTime, int NoDataRec = 10, std::string name = "DATA");
    ~Recorder();

    void addToRec(int value);
    void addToRec(double value);
    void addToRec(double array[], int sizeofarray);
    void addToRec(std::array<double, 3> array);
    void addToRec(std::array<double, 6> array);
    void addToRec(std::array<double, 7> array);
    void addToRec(std::array<double, 16> array);


    void addToRec(Vector3d& vector);
    void addToRec(Vector2d& vector);
    void addToRec(Vector4d& vector);
    void addToRec(VectorXd& vector);
    void addToRec(Quaterniond& vector);

    void saveData();
    void next();

  private:
    int _index;
    int _columnindex;
    int _rowindex;
    double _t_rec;
    std::string _name;
    Matrix<double, Dynamic, Dynamic> _DAT;
  };

  Recorder::Recorder(double t_rec, double sampleTime, int NoDataRec, std::string name) {
    _DAT.resize((int)(t_rec / sampleTime + 2), NoDataRec);
    _DAT.setZero();
    _rowindex = 0;
    _columnindex = 0;
    _t_rec = t_rec;
    _name = name;
  };
  Recorder::~Recorder() {
    saveData();
  };
  void Recorder::addToRec(int value) {
    _DAT(_rowindex, _columnindex) = value;
    _columnindex++;
  }
  void Recorder::addToRec(double value) {
    _DAT(_rowindex, _columnindex) = value;
    _columnindex++;
  }
  void Recorder::addToRec(double array[], int sizeofarray) {
    // cout << "TODO: size of array is manual" << endl;
    for (int i = 0; i < sizeofarray; i++) {
      _DAT(_rowindex, _columnindex) = array[i];
      _columnindex++;
    }
  };
  void Recorder::addToRec(std::array<double, 7> array) {
    // cout << "TODO: size of array is manual" << endl;
    for (int i = 0; i < 7; i++) {
      _DAT(_rowindex, _columnindex) = array[i];
      _columnindex++;
    }
  };

  void Recorder::addToRec(std::array<double, 16> array) {
    // cout << "TODO: size of array is manual" << endl;
    for (int i = 0; i < 16; i++) {
      _DAT(_rowindex, _columnindex) = array[i];
      _columnindex++;
    }
  };

  void Recorder::addToRec(std::array<double, 6> array) {
    // cout << "TODO: size of array is manual" << endl;
    for (int i = 0; i < 6; i++) {
      _DAT(_rowindex, _columnindex) = array[i];
      _columnindex++;
    }
  };
  void Recorder::addToRec(std::array<double, 3> array) {
    // cout << "TODO: size of array is manual" << endl;
    for (int i = 0; i < 3; i++) {
      _DAT(_rowindex, _columnindex) = array[i];
      _columnindex++;
    }
  };
  void Recorder::addToRec(Vector3d& vector) {
    for (int i = 0; i < vector.size(); i++) {
      _DAT(_rowindex, _columnindex) = vector[i];
      _columnindex++;
    }
  };
  void Recorder::addToRec(Vector2d& vector) {
    for (int i = 0; i < vector.size(); i++) {
      _DAT(_rowindex, _columnindex) = vector[i];
      _columnindex++;
    }
  };
  void Recorder::addToRec(Vector4d& vector) {
    for (int i = 0; i < vector.size(); i++) {
      _DAT(_rowindex, _columnindex) = vector[i];
      _columnindex++;
    }
  };
  void Recorder::addToRec(VectorXd& vector) {
    for (int i = 0; i < vector.size(); i++) {
      _DAT(_rowindex, _columnindex) = vector[i];
      _columnindex++;
    }
  };
  void Recorder::addToRec(Quaterniond& vector) {
    _DAT(_rowindex, _columnindex) = vector.x();
    _columnindex++;
    _DAT(_rowindex, _columnindex) = vector.y();
    _columnindex++;
    _DAT(_rowindex, _columnindex) = vector.z();
    _columnindex++;
    _DAT(_rowindex, _columnindex) = vector.w();
    _columnindex++;
  };

  void Recorder::saveData() {
    std::ofstream myfile;
    myfile.open(_name + ".m");
    myfile << _name << "m" <<"=[" << _DAT << "];\n";
    myfile.close();
    cout << "\n\n\t************Data was written successfully  ************\n";
  };
  void Recorder::next() {
    _rowindex++;
    _columnindex = 0;
  }
} // namespace panda_identification