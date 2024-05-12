#pragma once
#include <vector>


class NumberCheckerStructure
{
public:
	NumberCheckerStructure(std::vector<double>& input, const std::vector<double>& truthTbl)
		:truthTable(std::move(truthTbl))
	{
		inputForCheck.resize(input.size());  // Ensure copy has enough space

		std::copy(input.begin(), input.end(), inputForCheck.begin());
	}

	double* GetInput()
	{
		return inputForCheck.data();
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
	std::vector<double> inputForCheck;
};