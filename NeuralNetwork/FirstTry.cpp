#include <memory.h>
#include <opencv2/opencv.hpp>
#include "Game.h"
#include "ConvolutionaLayer.h"
#include "PoolingLayer.h"
#include <vector>
#include "ConvolutionalPartFull.h"
#include "Network.h"
#include "NumberCheckerStructure.h"
#include "Shuffler.h"
#include <string>

//std::vector<int> filters = { 4,3 };// number filters in each layer
//ConvolutionalPartFull conv = ConvolutionalPartFull(2, 28, 3, 2, filters);

//int convSize = conv.getSizeOutput();
//Network normalNN = Network({ conv.getSizeOutput(),10,5,10 });
//conv.LoadImage("D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//0//1.jpg");

/*Shuffler shuffler = Shuffler();

shuffler.createRepoSuffled("D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST-Shuffled");

std::vector<std::string> paths;
paths.resize(10);
paths[0] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//0";
paths[1] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//1";
paths[2] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//2";
paths[3] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//3";
paths[4] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//4";
paths[5] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//5";
paths[6] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//6";
paths[7] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//7";
paths[8] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//8";
paths[9] = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//9";

shuffler.shuffleFromMNISTjpg(paths, "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST-Shuffled");

int a;*/

//TrainSingleNum(200, conv, normalNN);

//TrainDataSet(conv, normalNN);


void TrainDataSet(ConvolutionalPartFull& conv, Network& normalNN)
{
    std::string path = "D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST-Shuffled";
    for (int i = 0; i < 1; i++)
    {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file()) {
                if (entry.path().extension() != ".txt")
                {
                    std::filesystem::path mainPath(path);
                    auto fullPath = mainPath / entry.path(); // This constructs the full path
                    conv.LoadImage(fullPath.string()); // Use the full path as a string
                }
                else
                {
                    std::ifstream file(entry.path()); // Open the file
                    int number;
                    std::vector<double> numberCheck(10, 0);
                    if (file.is_open()) {
                        std::string firstLine;
                        std::getline(file, firstLine); // Read the first line
                        file.close(); // Close the file
                        number = std::stoi(firstLine);

                        numberCheck[number] = 1.0;

                    }
                    else {
                        std::cerr << "Could not open file: " << entry.path() << std::endl;
                    }
                    std::vector<double> flat = conv.CalculateOutput();

                    auto result = normalNN.Classify(flat.data(), 10);

                    std::vector<NumberCheckerStructure> dataPoints;
                    NumberCheckerStructure pp = NumberCheckerStructure(flat, numberCheck);
                    dataPoints.push_back(pp);

                    normalNN.TrainEfficient(dataPoints, 0.01);

                    auto gradients = normalNN.returnGradientsInputLayer();

                    conv.TainEfficiently(gradients, 0.01);

                    normalNN.ClearGrads();
                }



            }
        }
    }

    conv.LoadImage("D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//0//1.jpg");
    std::vector<double> flat = conv.CalculateOutput();
    auto result = normalNN.Classify(flat.data(), 10);

    conv.LoadImage("D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//4//2.jpg");
    std::vector<double> flat1 = conv.CalculateOutput();
    auto result1 = normalNN.Classify(flat1.data(), 10);

    conv.LoadImage("D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//9//87.jpg");
    std::vector<double> flat2 = conv.CalculateOutput();
    auto result2 = normalNN.Classify(flat2.data(), 10);
    int a;
}

void TrainSingleNum(int numIters, ConvolutionalPartFull& conv, Network& normalNN)
{

    conv.LoadImage("D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//9//87.jpg");
    for (int i = 0; i < numIters; i++)
    {
        std::vector<double> flat = conv.CalculateOutput();

        auto result = normalNN.Classify(flat.data(), 10);

        std::vector<NumberCheckerStructure> dataPoints;
        NumberCheckerStructure pp = NumberCheckerStructure(flat, { 0,0,0,0,0,0,0,0,0,1 });
        dataPoints.push_back(pp);

        normalNN.TrainEfficient(dataPoints, 0.01);

        auto gradients = normalNN.returnGradientsInputLayer();

        conv.TainEfficiently(gradients, 0.01);

        normalNN.ClearGrads();
    }
    conv.LoadImage("D://C++//NeuralNetwork//MNIST Dataset JPG format//MNIST Dataset JPG format//MNIST - JPG - training//9//87.jpg");
    std::vector<double> flat2 = conv.CalculateOutput();
    auto result2 = normalNN.Classify(flat2.data(), 10);
    int a;
}