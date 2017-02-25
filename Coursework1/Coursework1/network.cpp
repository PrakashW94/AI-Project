#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include "network.h"

using namespace std;

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
	accuracy = 1;

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

	//cout << endl << "Network Initialised with "<< hiddenNodesCount << " hidden nodes." << endl;
}

//calculate the number of rows needed in each dataset 
//for each set, loop until you've filled the set
//taking a random index out of the index file in each pass

void Network::selectTraining(int type, vector<vector<vector<float>>> inputDataSet, int desiredPasses, int networkCount)
{
	switch (type)
	{
		case 1: //Split 60/20/20
		{
			passes = desiredPasses;
			float pastAcc;
			ofstream accTest;
			accTest.open("acctest.csv");
			accTest << "pass,msqer" << endl;
			for (int i = 1; i <= passes; i++)
			{
				if (i % 100 == 0)
				{
					pastAcc = accuracy;
					getOutput(inputDataSet[1], false);
					accTest << i << ", " << accuracy << endl;
					//cout << "Simulation running... " << endl << "Passes complete: " << i << endl << "Accuracy on validation set: " << accuracy << endl << endl;
					if (pastAcc < accuracy)
					{
						//cout << "Found minimum! Terminating network training..." << endl << endl;
						passes = i;
					}
				}
				if (i == passes)
				{//if last pass, create output files
					string filename = "network" + to_string(networkCount) + to_string(hiddenNodesCount);
					getOutput(inputDataSet[1], true, filename);
				}
				else
				{
					runOnce(inputDataSet[0], i, false);
				}
			}
			accTest.close();
			//cout << "Simulation Complete. " << endl;
			break;
		}

		case 2:
		{
			
			break;
		}

		case 3:
		{
			
			break;
		}
	}
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

//for each row in a given dataset, calculate a predicted output and output it
void Network::getOutput(vector<vector<float>> inputData, bool createOutput, string fileName)
{
	vector<float> predictedOutputAccuracy;
	int rowId = 1;

	ofstream outputFile;
	if (createOutput)
	{
		string fullFileName = "output/" + fileName + ".csv";
		outputFile.open(fullFileName);
		outputFile << "Row, Predicted, Correct" << endl;
	}

	for (vector<float> row : inputData)
	{
		forwardPass(row);

		//output results
		int outputNodeId = inputNodesCount + hiddenNodesCount + 1;
		float correctOutput = row.back();
		if (createOutput)
		{
			outputFile << rowId << ", " << getNodeById(outputNodeId).nodeOutput << ", " << correctOutput << endl;
		}
		predictedOutputAccuracy.push_back(correctOutput - getNodeById(outputNodeId).nodeOutput);
		rowId++;
	}
	if (createOutput)
	{
		outputFile.close();
	}
	calculateAccuracy(predictedOutputAccuracy);
}

//run for a single pass, loop is controlled externally
//allows for different(varying) datasets to be passed in
void Network::runOnce(vector<vector<float>> inputData, int loopCounter, bool createOutput)
{
	int rowId = 1;
	ofstream outputFile;
	if (createOutput)
	{
		outputFile.open("output.csv");
		outputFile << "Loop, Row, Predicted, Correct" << endl;
	}

	for (vector<float> row : inputData)
	{
		forwardPass(row);
		backwardPass(row);

		//output results
		if (createOutput)
		{//createOutput is set on the final pass through
			int outputNodeId = inputNodesCount + hiddenNodesCount + 1;
			float correctOutput = row.back();
			outputFile << loopCounter << ", " << rowId << ", " << getNodeById(outputNodeId).nodeOutput << ", " << correctOutput << endl;
		}
		rowId++;
	}
}

//forward pass for single row of data
void Network::forwardPass(vector<float> inputRow)
{
	//set input node values
	for (int i = 1; i <= inputNodesCount; i++)
	{
		nodeList[i - 1].setNodeOutput(inputRow[i - 1]);
	}

	//initialise loop variables
	int inputNodesLower = 1;
	int inputNodesUpper = inputNodesCount;
	int hiddenNodesLower = inputNodesCount + 1;
	int hiddenNodesUpper = inputNodesCount + hiddenNodesCount;
	int outputNodeId = inputNodesCount + hiddenNodesCount + 1;

	//calculate hidden node outputs
	for (int j = hiddenNodesLower; j <= hiddenNodesUpper; j++)
	{
		float hiddenNodeValue = 0;
		for (int i = inputNodesLower; i <= inputNodesUpper; i++)
		{
			hiddenNodeValue += getNodeById(i).nodeOutput * weightsMatrix[i][j];
		}
		hiddenNodeValue += getNodeById(j).bias * weightsMatrix[0][j];
		nodeList[j - 1].setNodeOutput(hiddenNodeValue);
	}

	//calculate output
	float output = 0;
	
	for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
	{
		output += getNodeById(i).nodeOutput * weightsMatrix[i][outputNodeId];
	}
	output += getNodeById(outputNodeId).bias * weightsMatrix[0][outputNodeId];
	nodeList[outputNodeId - 1].setNodeOutput(output);
}
//backward pass for single row of data
void Network::backwardPass(vector<float> inputRow)
{
	//initialise loop variables
	int inputNodesLower = 1;
	int inputNodesUpper = inputNodesCount;
	int hiddenNodesLower = inputNodesCount + 1;
	int hiddenNodesUpper = inputNodesCount + hiddenNodesCount;
	int outputNodeId = inputNodesCount + hiddenNodesCount + 1;
	
	//store correct output
	float correctOutput = inputRow.back();
	
	//backward pass
	//output node
	nodeList[outputNodeId - 1].setDeltaOutput(correctOutput);

	//hidden nodes
	for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
	{
		nodeList[i - 1].setDeltaHidden(weightsMatrix[i][outputNodeId], getNodeById(outputNodeId).delta);
	}

	//update weights
	//input layer -> hidden layer
	for (int i = inputNodesLower; i <= inputNodesUpper; i++)
	{
		for (int j = hiddenNodesLower; j <= hiddenNodesUpper; j++)
		{
			weightsMatrix[i][j] = weightsMatrix[i][j] + (stepParameter * getNodeById(j).delta * getNodeById(i).nodeOutput);
		}
	}

	//hidden layer -> output layer
	for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
	{
		weightsMatrix[i][outputNodeId] = weightsMatrix[i][outputNodeId] + (stepParameter * getNodeById(outputNodeId).delta * getNodeById(i).nodeOutput);
	}

	//bias weights
	for (int i = hiddenNodesLower; i <= outputNodeId; i++)
	{
		weightsMatrix[0][i] = weightsMatrix[0][i] + (stepParameter * getNodeById(i).delta * getNodeById(i).bias);
	}
}
//run for a given number of passes
/*
//input data is fixed for entire training
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
			forwardPass(row);
			backwardPass(row);

			int outputNodeId = inputNodesCount + hiddenNodesCount + 1;
			float correctOutput = row.back();
			//output results
			if (loop+1 % 500 == 0)
			{
				outputFile << loop << ", " << rowId << ", " << getNodeById(outputNodeId).nodeOutput << ", " << correctOutput << endl;
			}
			predictedOutputAccuracy.push_back(correctOutput - getNodeById(outputNodeId).nodeOutput);
			rowId++;
		}
		accuracyMatrix.push_back(predictedOutputAccuracy);
	}
	outputFile.close();
	cout << "Simulation Complete" << endl << endl;
}
*/
void Network::outputResults()
{
	cout << "RESULTS" << endl;
	cout << "Number of input nodes: " << inputNodesCount << endl;
	cout << "Number of hidden nodes: " << hiddenNodesCount << endl;
	cout << "Accuracy: " << accuracy << endl << endl;	
}

void Network::calculateAccuracy(vector<float> accuracyMatrix)
{
	float total = 0;
	for (float predictedOutput : accuracyMatrix)
	{
		total += pow(predictedOutput, 2);
	}
	accuracy = total / accuracyMatrix.size();
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