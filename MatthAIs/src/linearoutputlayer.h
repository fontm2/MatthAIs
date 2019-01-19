#ifndef LINEAROUTPUTLAYER_H
#define LINEAROUTPUTLAYER_H
#include <layer.h>

class LinearOutputLayer: public Layer
{

public:
    MatrixXd in;//layer input
    MatrixXd dF_T;//derivative of the activation functioin transposed
    MatrixXd delta_T;//collected error transposed
    MatrixXd weights;//ingoing weight matrix
    const std::shared_ptr<Layer> previous;
    LinearOutputLayer(int dim, const std::shared_ptr<Layer> previous, ActivationFunction actfunction, double learningrate_ = 0.01, double l2_reg = 0.01):previous(std::move(previous))
    {
        learningrate = learningrate_;
        l2_reg = l2_reg;
        activationfunction = actfunction;
        in.resize(previous->out.rows(), dim);
        //Weights = MatrixXd::Random(Previous->Out.cols(), dim);
        weights = MatrixXd::Zero(previous->out.cols(), dim);//bias is included in previous.Out.cols()
        //let bias_weight be initialized with zero
        weights.block(0,0,weights.rows()-1, weights.cols()) = MatrixXd::Random(weights.rows()-1, dim);
        out.resize(in.rows(), in.cols());
        delta_T.resize(in.cols(), in.rows());
        dF_T.resize(in.cols(), in.rows());
    }
    ~LinearOutputLayer() override {}
    void backPropagate(MatrixXd errors)
    {
        delta_T = dF_T.cwiseProduct(errors);
    }
    void forewardPropagate()
    {
        in = previous->out*weights;
        //Out = In.unaryExpr(ptr_fun(Layer::Activation));
        for(int i = 0; i < in.rows(); i++)
        {
            for(int j = 0; j < in.cols(); j++)
            {
                out(i,j) = Layer::activation(in(i,j));
            }
        }
        if(activationfunction == ActivationFunction::Sigmoid)
        {
            //dF_T = Out.unaryExpr(ptr_fun(Layer::Activation)).transpose();
            for(int i = 0; i < in.cols(); i++)
            {
                for(int j = 0; j < in.rows(); j++)
                {
                    dF_T(i,j) = Layer::derivative(out(j,i));
                }
            }
        }
        //for ReLU and leakyReLU
        else
        {
            //dF_T = In.unaryExpr(ptr_fun(Layer::Activation)).transpose();
            for(int i = 0; i < in.cols(); i++)
            {
                for(int j = 0; j < in.rows(); j++)
                {
                    dF_T(i,j) = Layer::derivative(in(j,i));
                }
            }
        }
    }
    void updateWeights()
    {
        MatrixXd delta_Weights(weights.rows(),weights.cols());
        delta_Weights << (delta_T * previous->out).transpose();
//        Weights = Weights-learningrate / In.rows()*delta_Weights;
        //no l2 reg in bias_weight whicht is located at bottom row
        weights.block(0,0, weights.rows()-1, weights.cols()) =
                weights.block(0,0, weights.rows()-1, weights.cols()) -
                (learningrate / in.rows() * delta_Weights.block(0,0, weights.rows()-1, weights.cols()) +
                 learningrate / in.rows() * l2_reg * delta_Weights.block(0,0, weights.rows()-1, weights.cols()));
        //update weights for bias
        weights.row(weights.rows()-1) = weights.row(weights.rows()-1) -learningrate / in.rows() * delta_Weights.row(delta_Weights.rows()-1);

    }

private:

};

#endif // LINEAROUTPUTLAYER_H
