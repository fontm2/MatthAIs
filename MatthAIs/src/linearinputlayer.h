#ifndef LINEARINPUTLAYER_H
#define LINEARINPUTLAYER_H
#include <layer.h>

using namespace Eigen;

class LinearInputLayer: public Layer
{

public:
    LinearInputLayer(MatrixXd Input)
    {
        activationfunction = ActivationFunction::None;
        out.resize(Input.rows(), Input.cols()+1);
        VectorXd Bias = VectorXd::Ones(Input.rows());
        out << Input, Bias;
    }
    ~LinearInputLayer() override {}

private:


};

#endif // LINEARINPUTLAYER_H
