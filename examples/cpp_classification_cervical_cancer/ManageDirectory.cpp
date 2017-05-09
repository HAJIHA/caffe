#include "ManageDirectory.h"
#include "ConvertString.h"
#include "StringParse.h"


std::vector<std::string> CManageDirectory::get_files_inDirectory(const std::string& _path, const std::string& _filter)
{
	std::string searching = _path + _filter;

	std::vector<std::string> return_;

	_finddata_t fd;
	long handle = _findfirst(searching.c_str(), &fd);  //현재 폴더 내 모든 파일을 찾는다.

	if (handle == -1)    return return_;

	int result = 0;
	do
	{
		return_.push_back(fd.name);
		result = _findnext(handle, &fd);
	} while (result != -1);

	_findclose(handle);

	return return_;
}


std::vector<std::string> CManageDirectory::get_files_inDirectory(const std::string& _path, const std::string& _filter, int nDelTimeIntVal)
{
	std::string searching = _path + _filter;

	std::vector<std::string> return_;

	_finddata_t fd;
	long handle = _findfirst(searching.c_str(), &fd);  //현재 폴더 내 모든 파일을 찾는다.

	if (handle == -1)    return return_;

	int result = 0;
	do
	{
		string strFullPath = _path + fd.name;
		bool bInterval = GetTimeIntervalCheck(CConvertString::StringToCString(strFullPath), nDelTimeIntVal);

		if (bInterval == false)
		{
			remove(strFullPath.c_str());
		}
		else
		{
			return_.push_back(fd.name);
		}
		result = _findnext(handle, &fd);
	} while (result != -1);

	_findclose(handle);

	return return_;
}

std::vector<std::string> CManageDirectory::get_files_inDirectoryIncludePath(const std::string& _path, const std::string& _filter, int nDelTimeIntVal)
{
	std::string searching = _path + _filter;

	std::vector<std::string> return_;

	_finddata_t fd;
	long handle = _findfirst(searching.c_str(), &fd);  //현재 폴더 내 모든 파일을 찾는다.

	if (handle == -1)    return return_;

	int result = 0;
	do
	{
		string strFullPath = _path + fd.name;
		bool bInterval = GetTimeIntervalCheck(CConvertString::StringToCString(strFullPath), nDelTimeIntVal);

		if (bInterval == false)
		{
			remove(strFullPath.c_str());
		}
		else
		{
			return_.push_back(strFullPath);
		}
		result = _findnext(handle, &fd);
	} while (result != -1);

	_findclose(handle);

	return return_;
}

std::vector<std::string> CManageDirectory::get_files_inDirectory(vector<std::string> vPath, const std::string& _filter, int nDelTimeIntVal)
{
	std::vector<std::string> return_;

	for (int i = 0; i < vPath.size(); i++)
	{
		vector<string> vStrTemp;
		vStrTemp = get_files_inDirectoryIncludePath(vPath[i], _filter, nDelTimeIntVal);
		return_.insert(return_.end(), vStrTemp.begin(), vStrTemp.end());
	}

	return return_;
}

bool CManageDirectory::GetTimeIntervalCheck(CString strPath, int nIntervalTime)
{
	CFileFind finder;
	BOOL IsFind = finder.FindFile(strPath);

	CTime tmLast;
	CTime curTime = CTime::GetTickCount();

	CString strWriteTime;
	if (IsFind)
	{
		finder.FindNextFileW();
		finder.GetLastWriteTime(tmLast);

		CTimeSpan elapsedTime = curTime - tmLast;
		int nDiffTime = elapsedTime.GetTotalHours();
		if (nDiffTime < nIntervalTime)
		{
			return true;//범위안
		}
		else
		{
			return false;
		}
	}
	return false;
}

int CManageDirectory::GetWriteTimeInterval(string strPath)
{
  CFileFind finder;
  BOOL IsFind = finder.FindFile(CConvertString::StringToCString(strPath));

  CTime tmLast;
  CTime cur = CTime::GetCurrentTime();
  CTimeSpan elapsedTime;
  if (IsFind)
  {
    finder.FindNextFileW();
    finder.GetLastWriteTime(tmLast);
    elapsedTime = cur - tmLast;
    return elapsedTime.GetTotalSeconds();
  }

  return -1;
}

CString CManageDirectory::GetWriteTime(CString strPath)
{
	CFileFind finder;
	BOOL IsFind = finder.FindFile(strPath);

	CTime tmLast;
	CString strWriteTime;
	if (IsFind)
	{
		finder.FindNextFileW();
		finder.GetLastWriteTime(tmLast);


		strWriteTime.Format(_T("%s"), tmLast.Format("%Y-%m-%d %H:%M:%S.000"));
	}
	return strWriteTime;
}



BOOL CManageDirectory::CreateDirectoryTree(LPCTSTR lptzPath)
{
	return CreateDirectoryTree(CConvertString::CStringToString(lptzPath));
}



BOOL CManageDirectory::CreateDirectoryTree(string strPath)
{
	vector<string> vPath = CStringParse::splitString(strPath, '\\');
	string strMerge;
	for (UINT i = 0; i<vPath.size(); i++)
	{
		strMerge += vPath[i];
		strMerge += "\\";
		CreateDirecotryStd(strMerge);
	}
	return TRUE;
}

bool CManageDirectory::CheckFileModeRW(string strFilePath)
{
	bool bRet = false;
	if ((_access(strFilePath.c_str(), 0)) != -1) //파일이 있으면
	{
		if ((_access(strFilePath.c_str(), 6)) != -1) //읽기쓰기 모두 가능하면
		{
			bRet = true;
		}
	}
	return bRet;
}


string CManageDirectory::GetCurrentDirString()
{
	TCHAR Path[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, Path);
	return CConvertString::TCHARToString(Path);
}

CString CManageDirectory::GetCurrentDirCString()
{
	TCHAR Path[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, Path);
	return Path;
}

BOOL CManageDirectory::CreateDirecotryStd(string strPath)
{
	CreateDirectory(CConvertString::StringToCString(strPath), NULL);
	return 0;
}


string CManageDirectory::checkNAddSlush(string strPath)
{
	int strlen = strPath.size();
	if (strPath[strlen - 1] == '\\' || strPath[strlen - 1] == '/')
	{
		return strPath;
	}
	else
	{
		return strPath + "/";
	}
}

string CManageDirectory::getFileName(string strFullPath)
{
	string strReturn = CStringParse::getLastString(strFullPath, '/');
	return strReturn;
}

string CManageDirectory::getFileName(string strFullPath, char delim)
{
  string strReturn = CStringParse::getLastString(strFullPath, delim);
  return strReturn;
}