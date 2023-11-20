#pragma once

#include <SFML/Graphics.hpp>
#include <memory>
class Graph
{
public:
	Graph(int width, int height)
		:
		width(width),
		height(height)
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
				/*float x = ((j / width) + 1) * width / 2;
				float y = ((-i / height) + 1) * height / 2;
				float res1 = j * weight1_1 + i * weight2_1 ;
				float res2 = j * weight1_2 + i * weight2_2 ;
				image->setPixel(y + bias1, x + bias2, res1 > res2 ? sf::Color::Cyan: sf::Color::Red);*/
				double y = (j)/(double)height;
				double x = (i) / (double)width;

				
				float res1 = y * weight1_1 + x * weight2_1 + bias1;
				float res2 = y * weight1_2 + x * weight2_2 + bias2;
				
				image->setPixel(j, i, res1> res2 ? sf::Color::Cyan : sf::Color::Red);

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

	float* getW11()
	{
		return &weight1_1;
	}
	float* getW12()
	{
		return &weight1_2;
	}
	float* getW21()
	{
		return &weight2_1;
	}
	float* getW22()
	{
		return &weight2_2;
	}

	float* Bias1()
	{
		return &bias1;
	}
	float* Bias2()
	{
		return &bias2;
	}


private:
	std::unique_ptr<sf::Image> image;
	std::unique_ptr<sf::Texture> texture;
	std::unique_ptr<sf::Sprite> sprite;
	float weight1_1 = 0.0f;
	float weight1_2 = 0.0f;
	float weight2_1 = 0.0f;
	float weight2_2 = 0.0f;
	float bias1 = 0.0f;
	float bias2 = 0.0f;
	int width;
	int height;
};