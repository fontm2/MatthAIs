#ifndef LINEARLAYER_H
#define LINEARLAYER_H
#include <layer.h>

class LinearLayer: public Layer
{

public:
    MatrixXd in;//layer input
    MatrixXd dF_T;//derivative of the activation functioin transposed
    MatrixXd delta_T;//collected error transposed
    MatrixXd weights;//ingoing weight matrix
    const std::shared_ptr<Layer> previous;
    LinearLayer(int dim, const std::shared_ptr<Layer> previous, ActivationFunction actfunction, double learningrate_ = 0.01, double l2_reg = 0.01):previous(std::move(previous))
    {
        learningrate = learningrate_;
        l2_reg = l2_reg;
        activationfunction = actfunction;
        in.resize(previous->out.rows(), dim);
        weights = MatrixXd::Zero(previous->out.cols(), dim);//bias is included in previous.Out.cols()
        //let bias_weight be initialized with zero
        weights.block(0,0,weights.rows()-1, weights.cols()) = MatrixXd::Random(weights.rows()-1, dim);//bias is included in previous.Out.cols()
        out.resize(in.rows(), in.cols()+1);
        delta_T.resize(in.cols(), in.rows());
        dF_T.resize(in.cols(), in.rows());
    }
    ~LinearLayer() override {}
    void backPropagate(MatrixXd OutgoingWeights, MatrixXd NextLayersDelta_T)
    {
        //remove weight from Bias. the weight beloning to the bias is located at the bottom row
        delta_T = dF_T.cwiseProduct(OutgoingWeights.block(0,0,OutgoingWeights.rows()-1,OutgoingWeights.cols())
                                    * NextLayersDelta_T);
    }
    void forewardPropagate()
    {
        in = previous->out*weights;
        //MatrixXd Temp = In.unaryExpr(ptr_fun(Layer::Activation));
        MatrixXd Temp(in.rows(), in.cols());
        for(int i = 0; i < in.rows(); i++)
        {
            for(int j = 0; j < in.cols(); j++)
            {
                Temp(i,j) = Layer::activation(in(i,j));
            }
        }
        VectorXd Bias = VectorXd::Ones(in.rows());
        out << Temp, Bias;
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


#endif // LINEARLAYER_H
