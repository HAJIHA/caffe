#include "StringParse.h"

vector<string>& CStringParse::split(const string &s, char delim, vector<string> &elems)
{
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim))
	{
		elems.push_back(item);
	}
	return elems;
}

vector<string> CStringParse::splitString(const string &s, char delim)
{
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

string CStringParse::getLastString(const string &s, char delim)
{
	vector<string> elems = splitString(s, delim);
	int nLast = elems.size() - 1;
	return elems[nLast];
}