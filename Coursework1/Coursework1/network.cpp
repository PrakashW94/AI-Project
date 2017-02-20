#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include "network.h"

float randomGen(int n)
{
	int min = int(-20000 / float(n));
	int max = int(20000 / float(n));
	return ((rand() % (max - min + 1) + min) / float(10000));
}

Network::Network(int iNodesCount, int hNodesCount)
{
	networkId = 0;

	inputNodesCount = iNodesCount;
	hiddenNodesCount = hNodesCount;
	//create input nodes
	//id is 1 to inputNodesCount (inclusive)
	//type is 1
	for (int i = 1; i <= inputNodesCount; i++)
	{
		Node node(i, 1, inputNodesCount);
		nodeList.push_back(node);
	}

	//create hidden nodes
	//id is inputNodesCount + 1 to inputNodesCount + hiddenNodesCount (inclusive)
	//type is 2
	for (int i = inputNodesCount+1; i <= (inputNodesCount + hiddenNodesCount); i++)
	{
		Node node(i, 2, inputNodesCount);
		nodeList.push_back(node);
	}

	//create output node
	//id is inputNodesCount + hiddenNodesCount + 1
	//type is 3
	Node node(inputNodesCount + hiddenNodesCount + 1, 3, 5);
	nodeList.push_back(node);

	//resize outer matrix
	weightsMatrix.resize(inputNodesCount + hiddenNodesCount + 2);

	for (int i = 0; i < inputNodesCount + hiddenNodesCount + 2; i++)
	{
		//resize innter matrix
		//makes access easier as indicies will reflect node connections
		//eg weight in cell [1][6] is weight for input from node 1 to node 6
		weightsMatrix[i].resize(inputNodesCount + hiddenNodesCount + 2);
		//initialise all values to 0
		weightsMatrix[i][0] = 0;
	}

	//populate weights matrix
	//input layer -> hidden layer weights
	for (int i = 1; i <= inputNodesCount; i++)
	{
		for (int j = inputNodesCount + 1; j <= inputNodesCount + hiddenNodesCount; j++)
		{
			weightsMatrix[i][j] = randomGen(inputNodesCount);
		}
	}
	
	//hidden layer -> output layer weights
	for (int i = inputNodesCount + 1; i <= inputNodesCount + hiddenNodesCount; i++)
	{
		weightsMatrix[i][inputNodesCount + hiddenNodesCount + 1] = randomGen(inputNodesCount);
	}

	//bias weights
	for (int i = inputNodesCount + 1; i <= inputNodesCount + hiddenNodesCount + 1; i++)
	{
		weightsMatrix[0][i] = randomGen(inputNodesCount);
	}

	cout << endl << "Network Initialised." << endl;
}

void Network::outputWeights()
{
	cout << "Weights Matrix" << endl;
	for (unsigned int i = 0; i < weightsMatrix.size(); i++)
	{
		cout << "Node " << i << ": ";
		for (unsigned int j = 0; j < weightsMatrix[i].size(); j++)
		{
			cout << weightsMatrix[i][j] << ", ";
		}
		cout << endl;
	}
	cout << endl;
}

Node Network::getNodeById(int id)
{
	return nodeList[id - 1];
}

void Network::run(vector<vector<float>> inputData, int desiredPasses)
{
	passes = desiredPasses;
	ofstream outputFile;
	outputFile.open("output.csv");
	outputFile << "Loop, Row, Predicted, Correct" << endl;
	for (int loop = 0; loop < passes; loop++)
	{
		if (loop % 500 == 0)
		{
			cout << "Simulation running, " << loop << " passes complete." << endl;
		}

		vector<float> predictedOutputAccuracy;

		int rowId = 1;
		for (vector<float> row : inputData)
		{
			//set input node values
			for (int i = 1; i <= inputNodesCount; i++)
			{
				nodeList[i - 1].setNodeOutput(row[i - 1]);
			}

			//store correct output
			float correctOutput = row.back();

			//initialise loop variables
			int inputNodesLower = 1;
			int inputNodesUpper = inputNodesCount;
			int hiddenNodesLower = inputNodesCount + 1;
			int hiddenNodesUpper = inputNodesCount + hiddenNodesCount;

			//calculate hidden node outputs
			for (int j = hiddenNodesLower; j <= hiddenNodesUpper; j++)
			{
				float hiddenNodeValue = 0;
				for (int i = inputNodesLower; i <= inputNodesUpper; i++)
				{
					hiddenNodeValue += getNodeById(i).getNodeOutput() * weightsMatrix[i][j];
				}
				hiddenNodeValue += getNodeById(j).getBias() * weightsMatrix[0][j];
				nodeList[j - 1].setNodeOutput(hiddenNodeValue);
			}

			//calculate output
			float output = 0;
			int outputNodeId = inputNodesCount + hiddenNodesCount + 1;
			for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
			{
				output += getNodeById(i).getNodeOutput() * weightsMatrix[i][outputNodeId];
			}
			output += getNodeById(outputNodeId).getBias() * weightsMatrix[0][outputNodeId];
			nodeList[outputNodeId - 1].setNodeOutput(output);

			//backward pass
			//output node
			nodeList[outputNodeId - 1].setDeltaOutput(correctOutput);

			//hidden nodes
			for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
			{
				nodeList[i - 1].setDeltaHidden(weightsMatrix[i][outputNodeId], getNodeById(outputNodeId).getDelta());
			}

			//update weights
			//input layer -> hidden layer
			for (int i = inputNodesLower; i <= inputNodesUpper; i++)
			{
				for (int j = hiddenNodesLower; j <= hiddenNodesUpper; j++)
				{
					weightsMatrix[i][j] = weightsMatrix[i][j] + (stepParameter * getNodeById(j).getDelta() * getNodeById(i).getNodeOutput());
				}
			}

			//hidden layer -> output layer
			for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
			{
				weightsMatrix[i][outputNodeId] = weightsMatrix[i][outputNodeId] + (stepParameter * getNodeById(outputNodeId).getDelta() * getNodeById(i).getNodeOutput());
			}

			//bias weights
			for (int i = hiddenNodesLower; i <= outputNodeId; i++)
			{
				weightsMatrix[0][i] = weightsMatrix[0][i] + (stepParameter * getNodeById(i).getDelta() * getNodeById(i).getBias());
			}

			//output results
			outputFile << loop << ", " << rowId << ", " << getNodeById(outputNodeId).getNodeOutput() << ", " << correctOutput << endl;
			predictedOutputAccuracy.push_back(correctOutput - getNodeById(outputNodeId).getNodeOutput());
			rowId++;
		}
		accuracyMatrix.push_back(predictedOutputAccuracy);
	}
	outputFile.close();
	calculateAccuracy();
	cout << "Simulation Complete" << endl << endl;
}

void Network::outputResults()
{
	cout << "RESULTS" << endl;
	cout << "Number of input nodes: " << inputNodesCount << endl;
	cout << "Number of hidden nodes: " << hiddenNodesCount << endl;
	cout << "Accuracy: " << accuracy << endl << endl;

	outputWeights();
}

void Network::calculateAccuracy()
{
	float total = 0;
	for (float predictedOutput : accuracyMatrix.back())
	{
		total += pow(predictedOutput, 2);
	}
	accuracy = total / accuracyMatrix.back().size();
}

void Network::setId(int id)
{
	networkId = id;
}

void Network::save(string filename)
{
	ofstream savefile;
	savefile.open("saved networks/" + filename + ".csv");
	savefile << "iNodes, " << inputNodesCount << endl;
	savefile << "hNodes, " << hiddenNodesCount << endl;
	savefile << "passes, " << passes << endl;
	savefile << "accuracy, " << accuracy << endl;
	for (vector<float> row : weightsMatrix)
	{
		for (float weight : row)
		{
			savefile << weight << ", ";
		}
		savefile << endl;
	}
	savefile.close();
}