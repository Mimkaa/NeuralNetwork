#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp> 
#include "NumberCheckerStructure.h"
#include "ConvPartVerTwo.h"

class ImageLoader
{
public:
    ImageLoader(const std::string& directory, const std::string& imageName) 
    : 
    directoryPath(directory) 
    {
        if (imageName == "all") {
            loadAllImages();
        }
        else if (isNumber(imageName)) {
            loadNumsImagesNames(directory, std::stoi(imageName));
        }

        else 
        {
            loadImage(imageName);
        }
    }

    bool isNumber(const std::string& str) {
        try {
            size_t pos;
            std::stoi(str, &pos);
            return pos == str.size();  // Check if all characters were used in conversion
        }
        catch (std::invalid_argument& e) {
            // Conversion failed because of non-numeric characters
            return false;
        }
        catch (std::out_of_range& e) {
            // Conversion failed because the number is out of the int range
            return false;
        }
    }

    void loadNumsImagesNames(const std::string& directory, int number)
    {
        namespace fs = std::filesystem;
        int count = 0; // Counter to track how many pairs we've added

        for (const auto& entry : fs::directory_iterator(directory)) {
            // Only proceed if we haven't reached the specified number of pairs
            if (count >= number) break;

            // Check if the file is an image and then find the corresponding text file
            if (entry.path().extension() == ".jpg") {
                std::string imagePath = entry.path().string();
                std::string textPath = imagePath.substr(0, imagePath.size() - 3) + "txt";

                if (fs::exists(textPath)) { // Ensure there is a corresponding text file
                    imagePaths.push_back(imagePath);
                    textPaths.push_back(textPath);
                    count++; // Increment our pair counter
                }
            }
        }
    }

    const std::vector<Eigen::MatrixXd> getByIndexImage(int index)
    {
        auto& imagePath = imagePaths[index];

        auto mat = getMatrixFromImage(imagePath);
        return { mat };
    }

    std::vector<double> getByindexNumber(int index)
    {
        auto& numberPath = textPaths[index];
        std::ifstream file(numberPath); // Open the file
        int number;
        std::vector<double> numberCheck(10, 0);
        if (file.is_open()) {
            std::string firstLine;
            std::getline(file, firstLine); // Read the first line
            file.close(); // Close the file
            number = std::stoi(firstLine);

            numberCheck[number] = 1.0;

        }
        return numberCheck;
    }

    std::vector<NumberCheckerStructure> makeNumberCheckerByIndexFC(int index)
    {
        auto& numberPath = textPaths[index];
        auto& imagePath = imagePaths[index];

        auto mat = getMatrixFromImage(imagePath);

        std::ifstream file(numberPath); // Open the file
        int number;
        std::vector<double> numberCheck(10, 0);
        if (file.is_open()) {
            std::string firstLine;
            std::getline(file, firstLine); // Read the first line
            file.close(); // Close the file
            number = std::stoi(firstLine);

            numberCheck[number] = 1.0;

        }
        std::vector<NumberCheckerStructure> dataPoints;
        NumberCheckerStructure pp = NumberCheckerStructure(mat.data(), numberCheck);
        dataPoints.push_back(pp);


        return dataPoints;
    }

    std::vector<NumberCheckerStructure> makeNumberCheckerByIndex(int index, ConvPartVerTwo& convPart)
    {
        auto& numberPath = textPaths[index];
        auto& imagePath = imagePaths[index];

        auto mat = getMatrixFromImage(imagePath);

        std::ifstream file(numberPath); // Open the file
        int number;
        std::vector<double> numberCheck(10, 0);
        if (file.is_open()) {
            std::string firstLine;
            std::getline(file, firstLine); // Read the first line
            file.close(); // Close the file
            number = std::stoi(firstLine);

            numberCheck[number] = 1.0;

        }
        std::vector<Eigen::MatrixXd> matricies{mat};
        convPart.forwardPass(matricies);

        std::vector<NumberCheckerStructure> dataPoints;
        NumberCheckerStructure pp = NumberCheckerStructure(convPart.getFlat().data(), numberCheck);
        dataPoints.push_back(pp);


        return dataPoints;

    }

    Eigen::MatrixXd& getFirst()
    {
        return images[0];
    }

    Eigen::MatrixXd& getLast()
    {
        return images[images.size()-1];
    }

    Eigen::MatrixXd& getByIndex(int index)
    {
        return images[index];
    }

   

    void loadAllImages() 
    {
        namespace fs = std::filesystem;
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") { // Checks for JPEG images
                loadSingleImage(entry.path().string());
            }
        }
    }

    void loadImage(const std::string& imageName) {
        std::string fullPath = directoryPath + "/" + imageName;
        loadSingleImage(fullPath);
    }

    

    void loadSingleImage(const std::string& imagePath) 
    {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            images.push_back(convertToEigen(image));
            std::cout << "Loaded image from: " << imagePath << std::endl;
        }
        else {
            std::cout << "Failed to load image from: " << imagePath << std::endl;
        }
    }

    Eigen::MatrixXd getMatrixFromImage(const std::string& imagePath)
    {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            std::cout << "Loaded image from: " << imagePath << std::endl;
            return convertToEigen(image);
        }
        else {
            std::cout << "Failed to load image from: " << imagePath << std::endl;
        }
    }

    Eigen::MatrixXd convertToEigen(const cv::Mat& image) 
    {
        Eigen::MatrixXd eigenImage(image.rows, image.cols);

        //Eigen::MatrixXd result = eigenImage.cast<double>();
        cv::cv2eigen(image, eigenImage);
        eigenImage = eigenImage / 255.0;
        return eigenImage;
    }

private:
	std::vector<Eigen::MatrixXd> images;
	std::string directoryPath;
    std::vector<std::string> imagePaths;
    std::vector<std::string> textPaths;
};