#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm> // For std::shuffle
#include <random> // For std::mt19937 and std::random_device
#include <numeric>


class Shuffler {
public:

    static void shuffleFromMNISTjpg(const std::vector<std::string>& dirPaths, const std::string& destination) {
        std::vector<std::filesystem::path> imagePathsNotShuffled;
        std::vector<int> numberNotShuffled;
        for (const auto& path : dirPaths) {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                if (entry.is_regular_file()) {
                    imagePathsNotShuffled.push_back(entry.path());
                    int num = path.back() - '0'; // Assuming single-digit number at the end of path
                    numberNotShuffled.push_back(num);
                }
            }
        }

        // Initialize indices with 0, 1, ..., n-1
        std::vector<int> indices(imagePathsNotShuffled.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Shuffle the indices
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<std::filesystem::path> imagePaths(imagePathsNotShuffled.size());
        std::vector<int> numbers(imagePathsNotShuffled.size());

        for (size_t i = 0; i < imagePathsNotShuffled.size(); i++) {
            imagePaths[i] = imagePathsNotShuffled[indices[i]];
            numbers[i] = numberNotShuffled[indices[i]];
        }

        std::filesystem::path destPath(destination);
        if (!std::filesystem::exists(destPath)) {
            std::filesystem::create_directories(destPath);
        }

        std::error_code ec;
        for (size_t i = 0; i < imagePaths.size(); i++) {
            std::filesystem::path newDestination = destPath / imagePaths[i].filename();
            std::filesystem::copy_file(imagePaths[i], newDestination, std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                std::cerr << "Error copying file: " << imagePaths[i] << " to " << newDestination << ", " << ec.message() << std::endl;
            }
            else {
                std::filesystem::path textFilePath = newDestination;
                textFilePath.replace_extension(".txt");
                std::ofstream textFile(textFilePath);
                if (textFile) {
                    textFile << numbers[i];
                }
                else {
                    std::cerr << "Error creating text file: " << textFilePath << std::endl;
                }
            }
        }
    }

    static void createRepoSuffled(const std::string& dirPath) {
        // Using the filesystem namespace directly without aliasing it inside the function
        try {
            // Check if the directory exists
            if (!std::filesystem::exists(dirPath)) {
                // The directory does not exist, so create it.
                std::filesystem::create_directory(dirPath);
                std::cout << "Directory created: " << dirPath << std::endl;
            }
            else if (!std::filesystem::is_directory(dirPath)) {
                // The path exists, but it's not a directory (could be a file)
                std::cerr << "The path exists but is not a directory: " << dirPath << std::endl;
            }
            else {
                // The directory already exists
                std::cout << "Directory already exists: " << dirPath << std::endl;
            }
        }
        catch (const std::filesystem::filesystem_error& e) {
            // Handle filesystem errors (e.g., permissions issues)
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
        catch (const std::exception& e) {
            // Handle any other exceptions
            std::cerr << "Standard exception: " << e.what() << std::endl;
        }
    }
};