#pragma once

#include <SFML/Graphics.hpp>
#include <memory>
#include "Network.h"
class Graph
{
public:
	Graph(int width, int height)
		:
		width(width),
		height(height),
		nn({2,3,2})
	{
		image = std::unique_ptr<sf::Image>(new sf::Image());
		image->create(width, height, sf::Color::Black);
		image->setPixel(20, 20, sf::Color::Cyan);

		texture = std::unique_ptr<sf::Texture>(new sf::Texture());
		texture->loadFromImage(*image);

		sprite = std::unique_ptr<sf::Sprite>(new sf::Sprite(*texture));

		

	}

	sf::Sprite& GetSprite()
	{
		return *sprite;
	}

	void Visualize()
	{
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				
				double input[2];
				input[0] = double(height - j - 1)/height;
				input[1] = double(i) / width;
				int res = nn.Classify(input);
				image->setPixel(height - j - 1, i, res>0.01?sf::Color::Cyan: sf::Color::Red);
				

			}
		}
	}

	void Update()
	{
		image->create(width, height, sf::Color::Black);
		Visualize();
		texture->loadFromImage(*image);
		sprite->setTexture(*texture);
	}

	void ImGuiStuff()
	{
		nn.ImguiStuff();
	}


private:
	std::unique_ptr<sf::Image> image;
	std::unique_ptr<sf::Texture> texture;
	std::unique_ptr<sf::Sprite> sprite;
	Network nn;
	int width;
	int height;
};