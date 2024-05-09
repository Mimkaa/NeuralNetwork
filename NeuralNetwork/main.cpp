#include <memory.h>
#include <opencv2/opencv.hpp>
#include "Game.h"
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "ImageLoader.h"
#include "PoolLayerVerTwo.h"
#include "ConvLayerVerTwo.h"
#include "MatrixShower.h"
#include "ConvPartVerTwo.h"
#include "Network.h"


void train(int numberImages, Network& fullyConnectedPart, ConvPartVerTwo& convPart, ImageLoader& il)
{
    for (int i = 0; i < numberImages; i++)
    {
        auto datapoints = il.makeNumberCheckerByIndex(i, convPart);
        fullyConnectedPart.TrainEfficient(datapoints, 0.01);
        auto gradients = fullyConnectedPart.returnGradientsInputLayer();
        fullyConnectedPart.ClearGrads();


        convPart.acceptGradients(gradients);
        convPart.backwardPass(0.01);
        
    }

}

void trainFullyConnected(int numberImages, Network& fullyConnectedPart, ImageLoader& il)
{
    for (int i = 0; i < numberImages; i++)
    {
        auto datapoints = il.makeNumberCheckerByIndexFC(i);
        fullyConnectedPart.TrainEfficient(datapoints, 0.01);
        auto gradients = fullyConnectedPart.returnGradientsInputLayer();
        fullyConnectedPart.ClearGrads();


       

    }
}

std::vector<int> guess(int imageIndex, Network& fullyConnectedPart, ConvPartVerTwo& convPart, ImageLoader& il)
{
    convPart.forwardPass(il.getByIndexImage(imageIndex));
    auto ress = fullyConnectedPart.Classify(convPart.getFlat().data(), 10);
    auto maxIt = std::max_element(ress.begin(), ress.end());
    int maxIndex = std::distance(ress.begin(), maxIt);
    

    auto correct = il.getByindexNumber(imageIndex);
    auto maxIt1 = std::max_element(correct.begin(), correct.end());
    int maxIndex1 = std::distance(correct.begin(), maxIt1);
    
    return { maxIndex, maxIndex1 };
}

void storeState(Network& fullyConnectedPart, ConvPartVerTwo& convPart)
{
    convPart.storeState("ConvPart.txt");
    fullyConnectedPart.storeState("FullyConnected.txt");
}

void loadState(Network& fullyConnectedPart, ConvPartVerTwo& convPart)
{
    convPart.LoadState("ConvPart.txt");
    fullyConnectedPart.loadState("FullyConnected.txt");
}


int main()
{
    

    int numberImages = 300;
    ImageLoader il = ImageLoader("D:/C++/NeuralNetwork/MNIST Dataset JPG format/MNIST Dataset JPG format/MNIST-Shuffled", std::to_string(numberImages));

    ConvPartVerTwo convPart = ConvPartVerTwo({ 8, 16 }, 28);
    Network fullyConnectedPart = Network({ convPart.getSizeFlattendOutput(),64,32,10});
    //loadState(fullyConnectedPart, convPart);
    train(numberImages, fullyConnectedPart, convPart, il);

    
    auto ans = guess(167, fullyConnectedPart, convPart, il);

    //storeState(fullyConnectedPart, convPart);
    
     return 0;
}