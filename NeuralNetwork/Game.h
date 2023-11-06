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

	}
    virtual ~Game()
    {
        delete[] MeanFPS;
    }
    void draw()
    {
        sf::CircleShape shape(100.f);
        shape.setFillColor(sf::Color::Green);
        window.draw(shape);
    }

    void Update()
    {
        
        
        
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
        }
        ImGui::Begin("Hello, world!");
        ImGui::Button("Look at this pretty button");
        ImGui::End();
       
        window.clear();
        draw();
        ImGui::SFML::Render(window);
        window.display();

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
            
            

            
            // this thing will update it onece per frame, and if lag is too big it will update everything a couple of times to catch up
            while (lag > FPSMCS) 
            {

                Update();
                lag -= FPSMCS;
                
            }
            // and only after that draw
            RenderAndEvents(deltaClock);
            std::cout << frameRate << std::endl;
        
    
        }
        ImGui::SFML::Shutdown();
	}
private:
	sf::RenderWindow window;
    float frameRate;
    int* MeanFPS;
    int sampleMeanFPS;
    std::chrono::time_point<std::chrono::steady_clock> previous_time;
};