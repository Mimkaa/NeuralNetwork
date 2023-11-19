#pragma once
#include <string>
#include <SFML/Graphics.hpp>
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
#include "imgui.h"
#include "imconfig-SFML.h"
#include "imgui-SFML.h"
#include <SFML/Window/Mouse.hpp>
#include <SFML/Window/Keyboard.hpp>
#include "Graphh.h"
#include <vector>
#include "DataPoint.h"

class Game
{
public:
	Game(int wWindow, int hWindow, const std::string& title, int sampleMeanFPS = 20)
        :
        sampleMeanFPS(sampleMeanFPS)
	{
		window.create(sf::VideoMode(wWindow, hWindow), title);
        ImGui::SFML::Init(window);
        // initialize mean FPS array
        MeanFPS = new int[sampleMeanFPS];
        for (int i = 0; i < sampleMeanFPS; i++) {
            MeanFPS[i] = 0;
        }
        // initialize the graph
        graph = std::unique_ptr<Graph>(new Graph(wWindow, hWindow));

	}
    virtual ~Game()
    {
        delete[] MeanFPS;
    }
    void draw()
    {
        
        window.draw(graph->GetSprite());
        if (!dataPoints.empty())
        {
            for (auto& d : dataPoints)
            {
                d.Draw(window);
            }
        }
    }

    void Update()
    {
        
        graph->Update();
        if (!dataPoints.empty())
        {
            std::cout<<"Cost: " << graph->ShowCost(dataPoints) << std::endl;
          
        }
        if (lern)
        {
            graph->TrainNN(dataPoints, 0.03);
            //std::cout << "LERN" << std::endl;
        }
       
    }

    void RenderAndEvents(sf::Clock& clc)
    {
        // ok i have no idea how to separate this imgui stuff in different functions, so I will leave it be so for now(
        ImGui::SFML::Update(window, clc.restart());
        sf::Event event;
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed)
                window.close();
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
            {
                sf::Vector2i position = sf::Mouse::getPosition(window);
                dataPoints.push_back(DataPoint(position.x, position.y, true));
            }
            if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
            {
                sf::Vector2i position = sf::Mouse::getPosition(window);
                dataPoints.push_back(DataPoint(position.x, position.y, false));
            }
            
            // Check if a key was pressed
            if (event.type == sf::Event::KeyPressed)
            {
                // Check which key was pressed
                if (event.key.code == sf::Keyboard::D)
                {
                    dataPoints.clear();
                }
                // Check which key was pressed
                if (event.key.code == sf::Keyboard::L)
                {
                    lern = !lern;
                }
                // Check which key was pressed
                if (event.key.code == sf::Keyboard::R)
                {
                    
                    double dist = 999999999999;
                    int indexToDelete = 0;
                    sf::Vector2i position = sf::Mouse::getPosition(window);
                    for (int i =0;i<dataPoints.size();i++)
                    {
                        double curDis = distance(position.x, dataPoints[i].x, position.y, dataPoints[i].y);
                        if (curDis <dist)
                        {
                            dist = curDis;
                            indexToDelete = i;
                        }
                    }
                    dataPoints.erase(dataPoints.begin() + indexToDelete);
                }
                
            }
            
        }
       
        graph->ImGuiStuff();
       
        window.clear();
        draw();
        ImGui::SFML::Render(window);
        window.display();

    }

    double distance(double x1, double x2, double y1, double y2)
    {
        double x = x2 - x1;
        double y = y2 - y1;
        return sqrt(x * x + y * y);
    }

	void run(int fps)
	{
        //Used to make the game framerate-independent.
        unsigned int lag = 0;

        //Get the current time and store it in a variable.
        previous_time = std::chrono::steady_clock::now();

        int FPSMCS = 1000000 / fps;

        
        sf::Clock deltaClock;
        while (window.isOpen())
        {
            unsigned int delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - previous_time).count();
            lag += delta_time;
            previous_time = std::chrono::steady_clock::now();
           

           
            // calculates the mean fps
            float FPS = fps;
            if (lag > FPSMCS)
            {
                FPS = 1000000.0f / lag;
            }

            for (int i = 0; i < sampleMeanFPS - 1; i++)
            {
                MeanFPS[i] = MeanFPS[i + 1];
            }
            MeanFPS[sampleMeanFPS-1] = FPS;

            int sum = 0;
            for (int i = 0; i < sampleMeanFPS; i++)
            {
                sum += MeanFPS[i];
            }
            frameRate = sum /= sampleMeanFPS;
            
            

            
            // this thing will update it once per frame, and if lag is too big it will update everything a couple of times to catch up
            while (lag > FPSMCS) 
            {

                Update();
                RenderAndEvents(deltaClock);
                lag -= FPSMCS;
                
            }
            
           
            //std::cout << frameRate << std::endl;
        
    
        }
        ImGui::SFML::Shutdown();
	}
private:
	sf::RenderWindow window;
    float frameRate;
    int* MeanFPS;
    int sampleMeanFPS;
    std::chrono::time_point<std::chrono::steady_clock> previous_time;
    std::unique_ptr<Graph> graph;
    std::vector<DataPoint> dataPoints;
    bool lern = false;
};
