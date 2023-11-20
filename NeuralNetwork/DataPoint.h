#pragma once
#include <SFML/Graphics.hpp>
#include <vector>

class DataPoint
{
public:
	DataPoint(double x, double y, bool correctness, int width, int height)
		:
		x(x),
		y(y),
		correct(correctness)
	{
		input.push_back(0);
		input.push_back(0);
		input[0] = (double)(x) / width;
		input[1] = (double)(y) / height;

		extected.push_back(0);
		extected.push_back(0);
		extected[1] = (double)(correct);
		extected[0] = (double)(!correct);
	}

	double* GetInput()
	{
		return input.data();
	}

	double* GetExpected()
	{
		return extected.data();
	}

	void Draw(sf::RenderWindow& window)
	{
		sf::CircleShape shape(10);
		shape.setFillColor(correct?sf::Color::Blue:sf::Color::Yellow);
		shape.setPosition(sf::Vector2f{(float)x, (float)y});
		window.draw(shape);
	}
public:
	std::vector<double> extected;
	std::vector<double> input;
	double x;
	double y;
	bool correct;
};