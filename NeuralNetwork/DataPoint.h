#pragma once
#include <SFML/Graphics.hpp>

class DataPoint
{
public:
	DataPoint(double x, double y, bool correctness)
		:
		x(x),
		y(y),
		correct(correctness)
	{}

	void Draw(sf::RenderWindow& window)
	{
		sf::CircleShape shape(10);
		shape.setFillColor(correct?sf::Color::Blue:sf::Color::Yellow);
		shape.setPosition(sf::Vector2f{(float)x, (float)y});
		window.draw(shape);
	}
public:
	double x;
	double y;
	bool correct;
};