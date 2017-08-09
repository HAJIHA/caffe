#pragma once
#include <vector>
#include <sstream>
#include <string>
using namespace std;
class CStringParse
{
private:
	static vector<string> &split(const string &s, char delim, vector<string> &elems);
public:
	static vector<string> splitString(const string &s, char delim);
	static string getLastString(const string &s, char delim);
};

