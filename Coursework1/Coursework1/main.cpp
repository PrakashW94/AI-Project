#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <windows.h>
#include <filesystem>

#include "network.h"

using namespace std;

vector<vector<float>> inputData;
vector<Network> networkList;
int inputSize;
string inputfile = "C:\\Users\\cgpw\\Desktop\\AI-Project\\Data\\CWDataStudentCleanOld.csv";
//string inputfile = "D:\\Work\\Part C\\Advanced AI\\Project\\Data\\CWDataStudentCleanOld.csv";

void readCSV()
{
	string line;
	ifstream csv(inputfile);
	float i;
	if (csv.is_open())
	{
		//first line has column headers
		getline(csv, line);
		while (getline(csv, line))
		{
			stringstream ss(line);
			vector<float> row;
			while (ss >> i)
			{
				row.push_back(i);
				if (ss.peek() == ',')
				{
					ss.ignore();
				}
			}
			inputData.push_back(row);
		}
		csv.close();
		inputSize = inputData.back().size() - 1;
	}
	else
	{
		cout << "Cannot open file!" << endl << endl;
	}
}

void standardiseData()
{
	vector<float> min;
	vector<float> max;

	for (float value : inputData[0])
	{
		min.push_back(value);
		max.push_back(value);
	}

	for (vector<float> row : inputData)
	{
		int i = 0;
		for (float value : row)
		{
			if (value < min[i])
			{
				min[i] = value;
			}
			if (value > max[i])
			{
				max[i] = value;
			}
			i++;
		}
	}

	for (unsigned int i = 0; i < inputData.size(); i++)
	{
		for (unsigned int j = 0; j < inputData[i].size(); j++)
		{
			inputData[i][j] = float(0.8) * (inputData[i][j] - min[j]) / (max[j] - min[j]) + float(0.1);
		}
	}
	cout << endl << inputData.size() << " Rows Standardised [0.1, 0.9]" << endl << endl;
}

void flushInputData()
{
	inputData.clear();
	cout << endl << "Data flushed." << endl;
}

void buildMenu()
{
	vector<string> commands =
	{
		"Read in and standardise data",
		"Run Neural Network Simulation",
		"Flush Networks"
	};
	int commandCount = commands.size();

	cout << "Menu" << endl << endl;
	int i = 1;
	for (string s : commands)
	{
		cout << i << ". " << s << endl;
		i++;
	}
	cout << "-1. Quit" << endl;
}

vector<vector<vector<float>>> splitInputDataKFolds(vector<vector<float>> inputData)
{
	//10 folds
	unsigned int foldCount = 10;
	
	//initialise result vector
	vector<vector<vector<float>>> result;

	//calculate size of each fold
	unsigned int foldSize = (int)(inputData.size()/ (float) foldCount);

	//initialise index
	int index;

	//initialise index vector
	vector<int> allIndex;
	for (unsigned int i = 0; i < inputData.size(); i++)
	{
		allIndex.push_back(i);
	}

	//shuffle indicies
	random_shuffle(allIndex.begin(), allIndex.end());

	//for each fold
	for (unsigned int i = 0; i < foldCount; i++)
	{
		vector<vector<float>> fold;

		//add foldSize rows of data
		for (unsigned int j = 0; j < foldSize; j++)
		{
			index = allIndex.back();
			allIndex.pop_back();

			fold.push_back(inputData[index]);
		}
		result.push_back(fold);
	}
	return result;
}

vector<vector<vector<float>>> splitInputDataStatic(vector<vector<float>> inputData)
{
	float trainingRatio = 0.6f;
	//initialise output vectors
	vector<vector<vector<float>>> result;
	vector<vector<float>> trainingData;
	vector<vector<float>> validationData;
	vector<vector<float>> testData;

	//initialise set sizes
	unsigned int trainingSize = (int)(inputData.size() * trainingRatio);
	unsigned int validationSize = (int)((inputData.size() - trainingSize) / (float)2);
	unsigned int testSize = validationSize;

	//initialise index
	int index;

	//initialise index vector
	vector<int> allIndex;
	for (unsigned int i = 0; i < inputData.size(); i++)
	{
		allIndex.push_back(i);
	}

	//fill training set
	for (unsigned int i = 0; i < trainingSize; i++)
	{
		random_shuffle(allIndex.begin(), allIndex.end());
		index = allIndex.back();
		allIndex.pop_back();

		trainingData.push_back(inputData[index]);
	}

	//fill validation set
	for (unsigned int i = 0; i < validationSize; i++)
	{
		random_shuffle(allIndex.begin(), allIndex.end());
		index = allIndex.back();
		allIndex.pop_back();

		validationData.push_back(inputData[index]);
	}

	//fill test set
	for (unsigned int i = 0; i < testSize; i++)
	{
		random_shuffle(allIndex.begin(), allIndex.end());
		index = allIndex.back();
		allIndex.pop_back();

		testData.push_back(inputData[index]);
	}

	//add all output vectors to one to be returned
	result.push_back(trainingData);
	result.push_back(validationData);
	result.push_back(testData);
	return result;
}

//denormalise using linear relationship
float denormaliseMSQE(float value)
{
	return (7401.5f*value + 15.318f);
}

int main()
{
	srand( unsigned int (time(NULL)));
	
	int menu = 0;
	while (1)
	{
		buildMenu();
		cin >> menu;

		switch (menu)
		{
			case 1: //read csv and standardise
			{
				readCSV();
				standardiseData();
				break;
			}

			case 2:
			{//train network
				int trainingType;
				cout << endl << "Select training method:" << endl << endl;
				cout << "1. Static 60/20/20." << endl;
				cout << "2. K-fold Cross Validation." << endl;
				cin >> trainingType;

				unsigned int hiddenNodesCount;
				unsigned int passes;
				cout << endl << "Enter the maximum number of hidden nodes to be used." << endl;
				cin >> hiddenNodesCount;
				cout << endl << "Enter the maximum number of passes the network should make." << endl;
				cin >> passes;

				bool bDriver;
				cout << endl << "Do you want to use a bold driver approach? (1/0)" << endl;
				cin >> bool(bDriver);

				cout << endl;
				switch (trainingType)
				{
					case 1:
					{
						for (unsigned int i = 2; i <= hiddenNodesCount; i++)
						{
							std::experimental::filesystem::create_directory("output/static/hn" + to_string(i));
							cout << "Simulation running... training networks with " << i << " hidden nodes." << endl;
							for (unsigned int j = 0; j < 10; j++)
							{
								vector<vector<vector<float>>> inputDataSet = splitInputDataStatic(inputData);
								Network network(inputSize, i);
								network.staticTrainingBD(inputDataSet, passes, j, bDriver);
								network.setId(networkList.size());
								networkList.push_back(network);
							}
						}
						cout << endl << "Simulation Complete." << endl;

						//output networks
						ofstream networkListOutput;
						networkListOutput.open("output/static/networkoutput.csv");
						networkListOutput << "NetworkId, hNodes, Passes, Validation Accuracy, Validation Accuracy (DN), Test Accuracy, Test Accuracy DN" << endl;
						for (Network network : networkList)
						{
							networkListOutput << network.networkId << ", " << network.hiddenNodesCount << ", " << network.passes << ", " << network.accuracy << ", " << denormaliseMSQE(network.accuracy) << ", " << network.testSetAccuracy << ", " << denormaliseMSQE(network.testSetAccuracy) << endl;
						}
						networkListOutput.close();
						break;
					}

					case 2:
					{
						for (unsigned int i = 2; i <= hiddenNodesCount; i++)
						{
							std::experimental::filesystem::create_directory("output/kfolds/hn" + to_string(i));
							cout << "Simulation running... training networks with " << i << " hidden nodes." << endl;
							for (unsigned int j = 0; j < 5; j++)
							{
								vector<vector<vector<float>>> inputDataSet = splitInputDataKFolds(inputData);
								std::experimental::filesystem::create_directory("output/kfolds/hn" + to_string(i) + "/n" + to_string(j));
								Network network(inputSize, i);
								network.kFoldsTraining(inputDataSet, passes, j, bDriver);
								network.setId(networkList.size());
								networkList.push_back(network);
							}
						}
						cout << endl << "Simulation Complete." << endl;

						//output networks
						ofstream networkListOutput;
						networkListOutput.open("output/kfolds/networkoutput.csv");
						networkListOutput << "NetworkId, hNodes, Validation Accuracy, Validation Accuracy (DN), P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, Total" << endl;
						for (Network network : networkList)
						{
							networkListOutput << network.networkId << ", " << network.hiddenNodesCount << ", " << network.accuracy << ", " << denormaliseMSQE(network.accuracy);
							for (int passes : network.kStepPasses)
							{
								networkListOutput << ", " << passes;
							}
							networkListOutput << ", " << network.totalPasses << endl;
						}
						networkListOutput.close();
						break;
					}
				}
				break;
			}

			case 3:
			{
				unsigned int numberOfNetworks = networkList.size();
				for (unsigned int i = 0; i < numberOfNetworks; i++)
				{
					networkList.pop_back();
				}
				break;
			}
			case -1:
			{
				return 0;
			}
		}
	}
}