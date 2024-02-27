#include <memory.h>
#include <opencv2/opencv.hpp>
#include "Game.h"
#include "ConvolutionaLayer.h"
#include "PoolingLayer.h"
#include <vector>
#include "ConvolutionalPartFull.h"
#include "Network.h"
#include "NumberCheckerStructure.h"

int main()
{
    std::vector<int> filters = { 4,3 };// number filters in each layer
    ConvolutionalPartFull conv = ConvolutionalPartFull(2, 28, 3, 2, filters);
    
    int convSize = conv.getSizeOutput();
    Network normalNN = Network({ conv.getSizeOutput(),10,5,10 });
    conv.LoadImage("D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//0//1.jpg");

    for (int i = 0; i < 300; i++)
    {
        std::vector<double> flat = conv.CalculateOutput();

        auto result = normalNN.Classify(flat.data(), 10);

        std::vector<NumberCheckerStructure> dataPoints;
        NumberCheckerStructure pp = NumberCheckerStructure(flat.data(), { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
        dataPoints.push_back(pp);

        normalNN.TrainEfficient(dataPoints, 0.01);

        auto gradients = normalNN.returnGradientsInputLayer();

        conv.TainEfficiently(gradients, 0.01);

        normalNN.ClearGrads();
    }

    std::vector<double> flat = conv.CalculateOutput();
    auto result = normalNN.Classify(flat.data(), 10);
    
    std::unique_ptr<Game> game = std::make_unique<Game>(800, 800, "NN");

    game->run(60);

    return 0;
}