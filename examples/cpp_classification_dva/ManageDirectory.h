#pragma once
#include <stdio.h>    
#include <io.h>
#include <vector>
#include <afx.h>
using namespace std;



class CManageDirectory
{
public:
	static vector<std::string> get_files_inDirectory(const std::string& _path, const std::string& _filter);
	static vector<std::string> get_files_inDirectory(const std::string& _path, const std::string& _filter, int nDelTimeIntVal);
	static vector<std::string> get_files_inDirectoryIncludePath(const std::string& _path, const std::string& _filter, int nDelTimeIntVal);
	static vector<std::string> get_files_inDirectory(vector<std::string> vPath, const std::string& _filter, int nDelTimeIntVal);
	static bool GetTimeIntervalCheck(CString strPath, int nIntervalTime);
	static CString GetWriteTime(CString strPath);


	static BOOL CreateDirectoryTree(LPCTSTR lptzPath);
	static BOOL CreateDirectoryTree(string strPath);
	static bool CheckFileModeRW(string strFilePath);
	static string GetCurrentDirString();
	static CString GetCurrentDirCString();

	static BOOL CreateDirecotryStd(string strPath);
	static string checkNAddSlush(string strPath);
	static string getFileName(string strFullPath);
};

