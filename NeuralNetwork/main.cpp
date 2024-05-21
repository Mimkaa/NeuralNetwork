
#define _CRT_SECURE_NO_WARNINGS
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

#include <memory.h>
#include <opencv2/opencv.hpp>
#include "Game.h"
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <thread>
#include <chrono>


#include <Eigen/Dense>
#include "ImageLoader.h"
#include "PoolLayerVerTwo.h"
#include "ConvLayerVerTwo.h"
#include "MatrixShower.h"
#include "ConvPartVerTwo.h"
#include "Network.h"

#include <cstdio>
#include <cstdlib>

void drawTraining(FILE* gnuplotPipe, std::vector<std::pair<double, double>>& dataPoints)
{
    // Clear the previous plot
    //fprintf(gnuplotPipe, "clear\n");

    // Re-plot the data points with lines
    fprintf(gnuplotPipe, "plot '-' with lines title 'training'\n");

    // Send the new data points to Gnuplot
    for (const auto& point : dataPoints) {
        fprintf(gnuplotPipe, "%f %f\n", point.first, point.second);
    }

    // End the data input
    fprintf(gnuplotPipe, "e\n");

    // Flush the pipe to ensure Gnuplot receives the data
    fflush(gnuplotPipe);

}

int countTrueValues(const std::vector<bool>& guessRate) {
    return std::count(guessRate.begin(), guessRate.end(), true);
}


void train(int numberImages, Network& fullyConnectedPart, ConvPartVerTwo& convPart, ImageLoader& il, FILE* gnuplotPipe)
{
    std::vector<std::pair<double, double>> cost;
    std::vector<bool> guessRate;
    for (int i = 0; i < numberImages; i++)
    {
        auto datapoints = il.makeNumberCheckerByIndex(i, convPart);
        fullyConnectedPart.TrainEfficient(datapoints, 0.01);
        auto gradients = fullyConnectedPart.returnGradientsInputLayer();
        fullyConnectedPart.ClearGrads();


        convPart.acceptGradients(gradients);
        convPart.backwardPass(0.01);
        cost.emplace_back(i, fullyConnectedPart.Cost(datapoints[0]));

        drawTraining(gnuplotPipe, cost);
        
        convPart.forwardPass((il.getByIndexImage(i)));
        auto ress = fullyConnectedPart.Classify(convPart.getFlat().data(), 10);
        auto maxIt = std::max_element(ress.begin(), ress.end());
        int maxIndex = std::distance(ress.begin(), maxIt);

        auto correct = il.getByindexNumber(i);
        auto maxIt1 = std::max_element(correct.begin(), correct.end());
        int maxIndex1 = std::distance(correct.begin(), maxIt1);

        //std::cout << "predicted: "<< maxIndex << std::endl;
        //std::cout << "correct: " << maxIndex1 << std::endl;

        if (maxIndex == maxIndex1) {
            //std::this_thread::sleep_for(std::chrono::seconds(2));
            guessRate.push_back(true);
        }
        else
        {
            guessRate.push_back(false);
        }
        std::cout << "guess Rate: " << float(countTrueValues(guessRate))/float(guessRate.size()) << std::endl;

    }

}


void trainFullyConnected(int numberImages, Network& fullyConnectedPart, ImageLoader& il, FILE* gnuplotPipe)
{
    std::vector<std::pair<double, double>> cost;
    for (int i = 0; i < numberImages; i++)
    {
        auto datapoints = il.makeNumberCheckerByIndexFC(i);
        fullyConnectedPart.TrainEfficient(datapoints, 0.01);
        auto gradients = fullyConnectedPart.returnGradientsInputLayer();
        fullyConnectedPart.ClearGrads();
        cost.emplace_back(i, fullyConnectedPart.Cost(datapoints[0]));

        drawTraining(gnuplotPipe, cost);
       

    }
}

std::vector<int> guess(int imageIndex, Network& fullyConnectedPart, ConvPartVerTwo& convPart, ImageLoader& il)
{
    convPart.forwardPass(il.getByIndexImage(imageIndex));
    convPart.print();
    //auto ppf = convPart.getFlat();
    auto ress = fullyConnectedPart.Classify(convPart.getFlat().data(), 10);
    auto maxIt = std::max_element(ress.begin(), ress.end());
    int maxIndex = std::distance(ress.begin(), maxIt);
    

    auto correct = il.getByindexNumber(imageIndex);
    auto maxIt1 = std::max_element(correct.begin(), correct.end());
    int maxIndex1 = std::distance(correct.begin(), maxIt1);
    
    return { maxIndex, maxIndex1 };
}

std::vector<int> guessFullyConnected(int imageIndex, Network& fullyConnectedPart, ImageLoader& il)
{
   
    auto ress = fullyConnectedPart.Classify((il.getByIndexImage(imageIndex)[0]).data(), 10);
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

void setGnuplotPath(const std::string& gnuplotPath) {
    const char* path = std::getenv("PATH");
    std::string newPath = std::string("PATH=") + gnuplotPath + ";" + (path ? path : "");
    _putenv(newPath.c_str());
}


int main()
{
    

    int numberImages = 30000;
    ImageLoader il = ImageLoader("D:/C++/NeuralNetwork/MNIST Dataset JPG format/MNIST Dataset JPG format/MNIST-Shuffled", std::to_string(numberImages));

    ConvPartVerTwo convPart = ConvPartVerTwo({ 4, 8 }, 28);
    Network fullyConnectedPart = Network({ convPart.getSizeFlattendOutput(),128,64,10});

    FILE* gnuplotPipe = popen("D:\\gnuplot\\gnuplot\\bin\\gnuplot.exe -persistent", "w");

    if (gnuplotPipe) {
        // Initial Gnuplot setup
        fprintf(gnuplotPipe, "set title 'Dynamic Plot from std::vector'\n");
        fprintf(gnuplotPipe, "set xlabel 'X-axis'\n");
        fprintf(gnuplotPipe, "set ylabel 'Y-axis'\n");
        fprintf(gnuplotPipe, "set grid\n");
    }

    //trainFullyConnected(numberImages, fullyConnectedPart, il, gnuplotPipe);

    ///loadState(fullyConnectedPart, convPart);
    train(numberImages, fullyConnectedPart, convPart, il, gnuplotPipe);

    //auto ans = guessFullyConnected(0, fullyConnectedPart, il);
    auto ans = guess(0, fullyConnectedPart, convPart, il);
    auto ans2 = guess(80, fullyConnectedPart, convPart, il);
    auto ans3 = guess(100, fullyConnectedPart, convPart, il);
    auto ans4 = guess(801, fullyConnectedPart, convPart, il);
    auto ans5 = guess(10, fullyConnectedPart, convPart, il);
    auto ans6 = guess(12, fullyConnectedPart, convPart, il);
    auto ans7 = guess(40, fullyConnectedPart, convPart, il);

    //storeState(fullyConnectedPart, convPart);

    pclose(gnuplotPipe);

    
    
     return 0;
}