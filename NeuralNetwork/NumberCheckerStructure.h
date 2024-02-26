#pragma once
#include <vector>


class NumberCheckerStructure
{
public:
	NumberCheckerStructure(double* inputs, const std::vector<double>& truthTbl)
		:truthTable(std::move(truthTbl))
	{
		input = inputs;
	}

	double* GetInput()
	{
		return input;
	}
	
	double* GetExpected()
	{
		return truthTable.data();
	}

	int SizeExpected()
	{
		return truthTable.size();
	}

private:
	double* input;
	std::vector<double> truthTable;
};