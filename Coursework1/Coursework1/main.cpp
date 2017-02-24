#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>


#include "network.h"

using namespace std;

vector<vector<float>> inputData;
vector<Network> networkList;
int inputSize;
int hiddenNodesCount;
unsigned int passes;
//string inputfile = "C:\\Users\\cgpw\\Desktop\\AI-Project\\Data\\CWDataStudentClean.csv";
string inputfile = "D:\\Work\\Part C\\Advanced AI\\Project\\Data\\CWDataStudentClean.csv";

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
		"Run Neural Network Simulation"
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
				cout << "2. K-fold Cross Validation (To do)" << endl;
				cin >> trainingType;

				cout << endl << "Enter the number of hidden nodes to be used." << endl;
				cin >> hiddenNodesCount;
				cout << endl << "Enter the number of passes the network should make." << endl;
				cin >> passes;

				Network network(inputSize, hiddenNodesCount);

				network.selectTraining(trainingType, inputData, passes);
				network.setId(networkList.size());
				networkList.push_back(network);
				break;
			}
			/*
			case 3: 
			{
				cout << endl << "Enter the number of hidden nodes to be used." << endl;
				cin >> hiddenNodesCount;
				cout << endl << "Enter the number of passes the network should make." << endl;
				cin >> passes;

				Network network(inputSize, hiddenNodesCount);
				for (unsigned int i = 0; i < passes; i++)
				{
					if (i+1 % 500 == 0)
					{
						cout << "Simulation running, " << i << " passes complete." << endl;
					}
					if (i == passes - 1)
					{
						network.runOnce(getDataset(inputData, 0.6f)[0], i, true);
					}
					else
					{
						network.runOnce(getDataset(inputData, 0.6f)[0], i, false);
					}
				}

				network.getOutput(getDataset(inputData, 0.4f)[0]);
				
				network.outputResults();
				network.setId(networkList.size());
				networkList.push_back(network);
				break;
			}

			case 4: 
			{
				cout << endl << "Enter the number of hidden nodes to be used." << endl;
				cin >> hiddenNodesCount;
				cout << endl << "Enter the number of passes the network should make." << endl;
				cin >> passes;

				Network network(inputSize, hiddenNodesCount);
				vector<vector<vector<float>>> dataSet = getDataset(inputData, 0.6f);
				network.run(dataSet[0], passes);
				network.getOutput(dataSet[1]);
				network.outputResults();
				network.setId(networkList.size());
				networkList.push_back(network);
				break;
			}

			case 5:
			{
				flushInputData();
				readCSV();
				standardiseData();
				break;
			}

			case 6:
			{
				for (Network network : networkList)
				{
					cout << "Network Id: " << network.networkId << endl;
					cout << "Number of input nodes: " << network.inputNodesCount << endl;
					cout << "Number of hidden nodes: " << network.hiddenNodesCount << endl;
					cout << "Passes: " << network.passes << endl;
					cout << "Accuracy: " << network.accuracy << endl << endl;
				}
				cout << endl << "Select a network to save to file." << endl;
				int selectedNetwork;
				cin >> selectedNetwork;
				cout << endl;

				string filename;
				cout << "Enter the desired filename." << endl;
				cin >> filename;
				cout << endl;
				networkList[selectedNetwork].save(filename);
				cout << "Selected network (Id = " << selectedNetwork << ") saved as " << filename << ".csv." << endl << endl;
				break;
			}
			*/
			case -1:
			{
				return 0;
			}
		}
	}
}