#pragma once
class Node
{
private:
	int nodeId;
	int nodeType;
	float nodeOutput;
	float bias;
	float delta;

	/*
	Notes:

	NodeId is a unique ID from 1 to inputSize + hiddenNodesCount + 1
	
	Node types
	1 - Input Node
	2 - Hidden Node
	3 - Output Node

	float is set to 1 during construction

	*/

public:
	//constructor
	Node(int id, int type, int inputCount);
	int getId();
	int getNodeType();
	float getBias();
	float getNodeOutput();
	void setNodeOutput(float value);
	float getDelta();
	void setDeltaOutput(float correctValue);
	void setDeltaHidden(float weight, float deltaOutput);
};