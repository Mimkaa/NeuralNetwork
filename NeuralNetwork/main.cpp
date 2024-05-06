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


int main()
{
    /*auto mm = il.getFirst();

    MatrixShower ms = MatrixShower(3, 14);

    
    
    ConvLayerVerTwo convL = ConvLayerVerTwo(1, 28, 3);
    convL.NyamImage(il.getByIndexVec(0));
    convL.convolve();

    
    PoolLayerVerTwo poolL = PoolLayerVerTwo(3, 28);
    poolL.pool(convL.getOutputs());

    ms.convertMatriciesToImages(poolL.getOutputs());
    ms.showImages();*/

    ImageLoader il = ImageLoader("D:/C++/NeuralNetwork/MNIST Dataset JPG format/MNIST Dataset JPG format/MNIST-Shuffled", "300");
    

    ConvPartVerTwo convPart = ConvPartVerTwo({ 4, 8 }, 28);
    Network fullyConnectedPart = Network({ convPart.getSizeFlattendOutput(),64,32,10});

    for (int i = 0; i < 300; i++)
    {
        auto datapoints = il.makeNumberCheckerByIndex(i, convPart);
        fullyConnectedPart.TrainEfficient(datapoints, 0.01);
        auto gradients = fullyConnectedPart.returnGradientsInputLayer();
        

        convPart.acceptGradients(gradients);
        convPart.backwardPass(0.01);
        fullyConnectedPart.ClearGrads();
    }

    

    convPart.forwardPass(il.getByIndexImage(175));
    auto ress = fullyConnectedPart.Classify(convPart.getFlat().data(), 10);
    auto maxIt = std::max_element(ress.begin(), ress.end());
    int maxIndex = std::distance(ress.begin(), maxIt);
    std::cout << maxIndex << std::endl;

    auto correct = il.getByindexNumber(175);
    auto maxIt1 = std::max_element(correct.begin(), correct.end());
    int maxIndex1 = std::distance(correct.begin(), maxIt1);
    std::cout << maxIndex1 << std::endl;
    
    return 0;
}