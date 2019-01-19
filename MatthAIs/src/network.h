#ifndef NETWORK_H
#define NETWORK_H
#include <linearlayer.h>
#include <linearinputlayer.h>
#include <linearoutputlayer.h>
#include <map>
#include <vector>
#include <string.h>




class Network : std::map<std::string, std::shared_ptr<Layer> >
{

public:
    MatrixXd Labels;
    int epoche{0};
    double l2_reg{0.001};
    std::vector<double> Hyperplanes;
    std::vector<std::pair<double,double> > PointsToDraw;
    Network(){}
    void init(MatrixXd Input, std::vector<int> layerdims, std::vector<ActivationFunction> layeractivations, double learningrate, double L2_reg)
    {
        l2_reg = L2_reg;
        MatrixXd In = Input.block(0,0,Input.rows(),Input.cols()-1);
        Labels = Input.col(Input.cols()-1);//last row of Input are labels
        //TODO: add one-hot-encoding for multiclass classification (if labels are 0,1,2,3,4... and not 0 or 1)
        for(int nbr = 0; nbr < layerdims.size(); nbr++)
        {
            if(nbr < 1 && layerdims.size()>1)//first hidden layer
            {
                //add inputlayer
                this->insert(std::make_pair(std::to_string(nbr) + "_InputLayer",
                                            std::shared_ptr<Layer>(new LinearInputLayer(In))));
                this->insert(std::make_pair(std::to_string(nbr+1) + "_Hiddenlayer_" + std::to_string(nbr+1),
                                            std::shared_ptr<Layer>(new LinearLayer(layerdims[nbr],
                                                                                   this->at(std::to_string(nbr) + "_InputLayer"),
                                                                                   layeractivations[nbr], learningrate, L2_reg))));
            }
            else if(nbr < 1 && layerdims.size() == 1)//inputlayer goes directly to output
            {
                //add inputlayer
                this->insert(std::make_pair(std::to_string(nbr) + "_InputLayer",
                                            std::shared_ptr<Layer>(new LinearInputLayer(In))));
                this->insert(std::make_pair(std::to_string(nbr+1) + "_OutputLayer",
                                            std::shared_ptr<Layer>(new LinearOutputLayer(layerdims[nbr],
                                                                                         this->at(std::to_string(nbr) + "_InputLayer"),
                                                                                         layeractivations[nbr], learningrate, L2_reg))));
            }
            else if(nbr < layerdims.size()-1)//for all intermedate layers
            {
                this->insert(std::make_pair(std::to_string(nbr+1) + "_Hiddenlayer_" + std::to_string(nbr+1),
                                            std::shared_ptr<Layer>(new LinearLayer(layerdims[nbr],
                                                                                   this->at(std::to_string(nbr) + "_Hiddenlayer_" + std::to_string(nbr)),
                                                                                   layeractivations[nbr], learningrate, L2_reg))));
            }
            else//for last layer
            {
                this->insert(std::make_pair(std::to_string(nbr+1) + "_OutputLayer",
                                            std::shared_ptr<Layer>(new LinearOutputLayer(layerdims[nbr],
                                                                                         this->at(std::to_string(nbr) + "_Hiddenlayer_" + std::to_string(nbr)),
                                                                                         layeractivations[nbr], learningrate, L2_reg))));
            }
        }
    }
    void activate()
    {
        std::map<std::string, std::shared_ptr<Layer> >::iterator begin = this->begin();
        begin++;//no activate on inputlayer
        std::map<std::string, std::shared_ptr<Layer> >::iterator end = this->end();
        for (std::map<std::string, std::shared_ptr<Layer> >::const_iterator it = begin; it != end; it++)
        {
            std::shared_ptr<LinearLayer> LinLay_ptr = std::dynamic_pointer_cast<LinearLayer>(it->second);
            if(LinLay_ptr != NULL)
            {
                LinLay_ptr->forewardPropagate();
            }
            std::shared_ptr<LinearOutputLayer> LinOutLay_ptr = std::dynamic_pointer_cast<LinearOutputLayer>(it->second);
            if(LinOutLay_ptr != NULL)
            {
                LinOutLay_ptr->forewardPropagate();
            }
        }
    }
    double backprop()
    {
        MatrixXd errors(this->at(std::to_string(this->size()-1) + "_OutputLayer")->out.rows() ,this->at(std::to_string(this->size()-1) + "_OutputLayer")->out.cols());
        errors = this->at(std::to_string(this->size()-1) + "_OutputLayer")->out - Labels;
        double error = (errors.cwiseProduct(errors)/2).sum();
        //Backprob error from output and label on Outputlayer
        std::map<std::string, std::shared_ptr<Layer> >::const_iterator outputlayer_it = this->end();
        outputlayer_it--;//end points behind the map
        std::shared_ptr<LinearOutputLayer> LinOutLay_ptr = std::dynamic_pointer_cast<LinearOutputLayer>(outputlayer_it->second);
        if(LinOutLay_ptr != NULL)
        {
            LinOutLay_ptr->backPropagate(errors.transpose());
        }

        //Backprop on Hiddenlayers. no backprop on inputlayer
        std::map<std::string, std::shared_ptr<Layer> >::iterator begin = this->end();
        begin--;//end points behind the map
        begin--;//the outpulayer has allready been processed
        std::map<std::string, std::shared_ptr<Layer> >::iterator end = this->begin();
        for (std::map<std::string, std::shared_ptr<Layer> >::const_iterator it = begin; it != end; it--)
        {
            std::map<std::string, std::shared_ptr<Layer> >::const_iterator itr_to_previous_layer = it;
            itr_to_previous_layer++;
            std::shared_ptr<LinearLayer> LinLay_ptr = std::dynamic_pointer_cast<LinearLayer>(it->second);
            if(LinLay_ptr != NULL)
            {
                std::shared_ptr<LinearOutputLayer> ptr_to_previous_LinOutLay = std::dynamic_pointer_cast<LinearOutputLayer>(itr_to_previous_layer->second);
                if(ptr_to_previous_LinOutLay != NULL)
                {
                    LinLay_ptr->backPropagate(ptr_to_previous_LinOutLay->weights, ptr_to_previous_LinOutLay->delta_T);
                }
                std::shared_ptr<LinearLayer> ptr_to_previous_LinLay = std::dynamic_pointer_cast<LinearLayer>(itr_to_previous_layer->second);
                if(ptr_to_previous_LinLay != NULL)
                {
                    LinLay_ptr->backPropagate(ptr_to_previous_LinLay->weights, ptr_to_previous_LinLay->delta_T);
                }
            }
        }
        //calculateing sum of all weights for L2_reg in log-likelihood
        double weight_sum_squarred = 0.0;
        begin = this->begin();
        begin++;//no weights on inputlayer
        end = this->end();
        for (std::map<std::string, std::shared_ptr<Layer> >::const_iterator it = begin; it != end; it++)//inputlayer contains no weights
        {
            //weigts corresponding to bias should not be summed up
            MatrixXd WeightsWithoutBias(0,0);
            std::shared_ptr<LinearLayer> LinLay_ptr = std::dynamic_pointer_cast<LinearLayer>(it->second);
            if(LinLay_ptr != NULL)
            {
                WeightsWithoutBias = LinLay_ptr->weights.block(0,0,LinLay_ptr->weights.rows()-1,LinLay_ptr->weights.cols());
                weight_sum_squarred +=(WeightsWithoutBias.cwiseProduct(WeightsWithoutBias)).sum();
            }
            std::shared_ptr<LinearOutputLayer> LinOutLay_ptr = std::dynamic_pointer_cast<LinearOutputLayer>(it->second);
            if(LinOutLay_ptr != NULL)
            {
                WeightsWithoutBias = LinOutLay_ptr->weights.block(0,0,LinOutLay_ptr->weights.rows()-1,LinOutLay_ptr->weights.cols());
                weight_sum_squarred +=(WeightsWithoutBias.cwiseProduct(WeightsWithoutBias)).sum();
            }

        }
        return (error + l2_reg * weight_sum_squarred / 2) / Labels.rows();
        //return error / Labels.rows();
    }
    void updateweights()
    {
        Hyperplanes.clear();
        PointsToDraw.clear();
        //inputlayer has no weights
        std::map<std::string, std::shared_ptr<Layer> >::iterator begin = this->begin();
        begin++;//no weights on inputlayer
        std::map<std::string, std::shared_ptr<Layer> >::iterator end = this->end();
        for (std::map<std::string, std::shared_ptr<Layer> >::const_iterator it = begin; it != end; it++)
        {
            std::shared_ptr<LinearLayer> LinLay_ptr = std::dynamic_pointer_cast<LinearLayer>(it->second);
            if(LinLay_ptr != NULL)
            {
                LinLay_ptr->updateWeights();
            }
            std::shared_ptr<LinearOutputLayer> LinOutLay_ptr = std::dynamic_pointer_cast<LinearOutputLayer>(it->second);
            if(LinOutLay_ptr != NULL)
            {
                LinOutLay_ptr->updateWeights();
                if(LinOutLay_ptr->weights.rows() == 3 && LinOutLay_ptr->weights.cols() ==1)
                {
                    Hyperplanes.push_back(LinOutLay_ptr->weights(0,0));
                    Hyperplanes.push_back(LinOutLay_ptr->weights(1,0));
                    Hyperplanes.push_back(LinOutLay_ptr->weights(2,0));
                }
            }
        }
        if(this->begin()->second->out.cols() == 577)
        {
            //fill the PointsToDraw vector for facedata.txt
            std::shared_ptr<LinearLayer> ptr_to_penultimateLayer =
                    std::dynamic_pointer_cast<LinearLayer>(this->at(std::to_string(this->size()-2) + "_Hiddenlayer_" + std::to_string(this->size()-2)));
            if(ptr_to_penultimateLayer != NULL)
            {
                for(auto idx : m_examples_to_display)
                {
                    PointsToDraw.push_back(std::make_pair(ptr_to_penultimateLayer->out(idx,0),ptr_to_penultimateLayer->out(idx,1)));
                }
            }
        }
        else
        {
            //fill the PointsToDraw vector for Patterns.txt with the first 10 examples
            std::shared_ptr<LinearLayer> ptr_to_penultimateLayer =
                    std::dynamic_pointer_cast<LinearLayer>(this->at(std::to_string(this->size()-2) + "_Hiddenlayer_" + std::to_string(this->size()-2)));
            if(ptr_to_penultimateLayer != NULL)
            {
                for(int idx = 0; idx < 10; idx++)
                {
                    //the first 5 are negative examples, the 5-10 are positive examples
                    PointsToDraw.push_back(std::make_pair(ptr_to_penultimateLayer->out(idx,0),ptr_to_penultimateLayer->out(idx,1)));
                }
            }
        }
    }

    double train()
    {
        activate();
        double error = backprop();
        updateweights();
        epoche++;
        return error;
    }

    template<typename M>
    M load_csv (const std::string & path) {
        std::ifstream indata;
        indata.open(path);
        std::string line;
        std::vector<double> values;
        uint rows = 0;
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }
        //RowMajor fills rows first
        MatrixXd res = Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
        return res;
    }


private:
    //at index: 0,1,2,4,6 we have negative examples
    //at index: 8,11,14,17,18 we have positive examples
    int m_examples_to_display[10] = {0,1,2,4,6,8,11,14,17,18};

};


#endif // NETWORK_H
