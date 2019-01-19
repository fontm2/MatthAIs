#include <QApplication>
#include "widget.h"
#include <QtPrintSupport>
#include <thread>

double ramp(double x)
{
  if (x > 0)
    return x;
  else
    return 0;
}

using namespace Eigen;

int main(int argc, char* argv[])
{
//    Network net;
//    vector<int> layerdims{2,1};
//    vector<ActivationFunction> layeractfunc{ActivationFunction::Sigmoid, ActivationFunction::Sigmoid};
////    MatrixXd Input(10,4);
////    VectorXd Labels(10);
////    Labels << 1, 0, 1, 1, 1, 0, 0, 0, 1, 0;
////    Input << MatrixXd::Random(10,3), Labels;
////    std::cout << Input << std::endl;
//    MatrixXd Input = net.load_csv<MatrixXd>("trainingdata.txt");
//    net.init(Input/255, layerdims, layeractfunc, 0.5, 0.5);
//    for(int i = 0; i < 100; i++)
//    {
//        std::cout << net.train() << std::endl;
//    }

//    ActivationFunction activationfunc = ActivationFunction::ReLU;
//    if(activationfunc == ActivationFunction::Sigmoid)
//    {
//            std::cout << "is sigmoid" << std::endl;
//    }
//    else
//    {
//             std::cout << "is not sigmoid" << std::endl;
//    }
//    Eigen::MatrixXd m = Eigen::MatrixXd::Random(4,3);
//    Eigen::VectorXd Bias = Eigen::VectorXd::Ones(m.rows());
//    Eigen::MatrixXd concat;
//    concat.resize(m.rows(),m.cols()+1);
//    concat << m, Bias;
//    MatrixXd mat(0,0);
//    mat =  concat.block(0,0,concat.rows(),concat.cols()-2);
//    MatrixXd mat_blocked = mat.block(0,0,concat.rows(),1);
//    std::cout << mat << std::endl;
//    mat.block(0,0,concat.rows(),1) = mat.block(0,0,concat.rows(),1).cwiseProduct(mat_blocked);
//    std::cout << mat << std::endl;
//    MatrixXd mat_T = mat.transpose();
//    MatrixXd processed = mat_T.unaryExpr(ptr_fun(ramp)).transpose();
//    std::cout << mat << std::endl;
//    std::cout << mat_blocked << std::endl;
//    std::cout << mat_T << std::endl;
//    std::cout << processed << std::endl;
    int test = 1;
    QApplication a(argc ,argv);
    Widget w;
    w.show() ;
    return a.exec();
}


