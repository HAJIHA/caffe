#pragma once
#include <afx.h>
#include <string>
#include <sstream>
using namespace std;
class CConvertString
{
public:
	CConvertString(void);
	~CConvertString(void);

	inline static string TCHARToString(const TCHAR* ptsz)
	{
		size_t len = wcslen((wchar_t*)ptsz);
		char* psz = new char[2*len + 1];

		size_t getVal = 0;
		wcstombs_s(&getVal, psz, 2*len + 1, ptsz, _TRUNCATE);

		string s = psz;
		delete [] psz;
		return s;
	}

	typedef basic_string<TCHAR> tstring;  

	inline static TCHAR* StringToTCHAR(string& s)
	{
		tstring tstr;
		const char* all = s.c_str();
		size_t len = 1 + strlen(all);
		wchar_t* t = new wchar_t[len]; 
		if (NULL == t) throw bad_alloc();
		//mbstowcs(t, all, len);
		size_t getVal = 0;
		mbstowcs_s(&getVal,t,len,all,len);
		return (TCHAR*)t;
	}

	inline static string CStringToString(CString cstrSrc)
	{
		return  TCHARToString((LPCTSTR)cstrSrc);
	}

	inline static CString StringToCString(string strSrc)
	{
		return CString(strSrc.c_str());
	}

	inline static string IntToString(const int nNum)
	{
		stringstream ss;
		ss << nNum;
		string s = ss.str();
		return s;
	}

	inline static CString IntToCString(const int nNum)
	{
		stringstream ss;
		ss << nNum;
		string s = ss.str();
		return StringToCString(s);
	}

	inline static long StringToLong(string strSrc)
	{
		return  static_cast<long>(atoi(strSrc.c_str()));
	}

	inline static int StringToInt(string strSrc)
	{
		return atoi(strSrc.c_str());
	}

	inline static double StringToDouble(string strSrc)
	{
		return atof(strSrc.c_str());
	}

	inline static CString DoubleToCString(double dValue)
	{
		CString strNum;
		strNum.Format(_T("%f"), dValue);
		return strNum;
	}

	inline static string DoubleToString(const double dValue)
	{	
		stringstream ss;
		ss << dValue;
		string s = ss.str();
		return s;
	}

	inline static char* CStringToCharP(CString strString)
	{
		string strStdTemp = CConvertString::CStringToString(strString);
		return _strdup(strStdTemp.c_str());
	}

	inline static char* StringToCharP(string strString)
	{
		return _strdup(strString.c_str());
	}

	inline static string replaceAll(const string &str, const string &pattern, const string &replace)   
	{   
		string result = str;   
		string::size_type pos = 0;   
		string::size_type offset = 0;   
		while((pos = result.find(pattern, offset)) != string::npos)   
		{   
			result.replace(result.begin() + pos, result.begin() + pos + pattern.size(), replace);   

			offset = pos + replace.size();   

		}   
		return result;   
	}


	inline static CString GetDbTime(const CString strTime)
	{
		CString strYYYY;
		CString strMM;
		CString strDD;
		CString strHH;
		CString strMinute;
		CString strSec;

		CString strRet;

		strYYYY = strTime.Left(4);
		strMM = strTime.Mid(4, 2);
		strDD = strTime.Mid(6, 2);

		strHH = strTime.Mid(8, 2);
		strMinute = strTime.Mid(10, 2);
		strSec = strTime.Mid(12, 2);

		strRet = strYYYY + _T("-");
		strRet += strMM + _T("-");
		strRet += strDD + _T(" ");

		strRet += strHH + _T(":");
		strRet += strMinute + _T(":");
		strRet += strSec;

		return strRet;
	}


	inline static string GetDbTime(const string strTime)
	{
		CString cstrTime = StringToCString(strTime);

		CString cstrCvt = GetDbTime(cstrTime);

		return CStringToString(cstrCvt);
	}

	inline static CString getCurTimeCString()
	{
		CString buffer, format;
		SYSTEMTIME cur_time;
		GetLocalTime(&cur_time);

		buffer.Format(_T("%04d-%02d-%02d %02d:%02d:%02d:%03ld\t"),
			cur_time.wYear,
			cur_time.wMonth,
			cur_time.wDay,
			cur_time.wHour,
			cur_time.wMinute,
			cur_time.wSecond,
			cur_time.wMilliseconds);

		return buffer;
	}
};

