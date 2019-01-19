#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <memory> // for unique_ptr

using namespace Eigen;

enum ActivationFunction{Sigmoid, ReLU, lReLU, None};

class Layer
{

public:
    double learningrate{0.001};
    double l2_reg{0.001};
    MatrixXd out;//layer output
    ActivationFunction activationfunction = ActivationFunction::None;//activation function
    Layer(){}
    Layer(MatrixXd Input){}//constructor for Inputlayer
    Layer(int dim, const Layer& previous, ActivationFunction actfunction, double learningrate_ = 0.01, double l2_reg = 0.01){}//Constructor for Hidden- and Outputlayers
    virtual ~Layer(){}

protected:
    double activation(double Input)
    {
        if(activationfunction == ActivationFunction::Sigmoid)
        {
            return 1 / (1 + std::exp(-Input));
        }
        else if(activationfunction == ActivationFunction::ReLU)
        {
            if(Input >= 0)
            {
                return Input;
            }
            else
            {
                return 0.0;
            }
        }
        else if(activationfunction == ActivationFunction::lReLU)
        {
            if(Input > 0)
            {
                return Input;
            }
            else
            {
                return 0.001*Input;
            }
        }
        //if ActivationFunction::None
        else
        {
            return Input;
        }
        return Input;
    }
    double derivative(double Input)
    {
        if(activationfunction == ActivationFunction::Sigmoid)
        {
            return Input * (1 - Input);
        }
        else if(activationfunction == ActivationFunction::ReLU)
        {
            if(Input > 0)
            {
                return 1.0;
            }
            else
            {
                return 0.0;
            }
        }
        else if(activationfunction == ActivationFunction::lReLU)
        {
            if(Input > 0)
            {
                return 1.0;
            }
            else
            {
                return 0.001;
            }
        }
        //if ActivationFunction::None
        else
        {
            return 1;
        }
    }
};


#endif // LAYER_H
