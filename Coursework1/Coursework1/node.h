#pragma once
class Node
{
public:
	/*
	NodeId is a unique ID from 1 to inputSize + hiddenNodesCount + 1

	Node types
	1 - Input Node
	2 - Hidden Node
	3 - Output Node

	bias is set to 1 during construction
	*/
	int nodeId;
	int nodeType;
	float nodeOutput;
	float bias;
	float delta;
	//constructor
	Node(int id, int type, int inputCount);




	void setNodeOutput(float value);
	void setDeltaOutput(float correctValue);
	void setDeltaHidden(float weight, float deltaOutput);
};