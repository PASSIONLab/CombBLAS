#include <ctime>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cout << "usage: " << endl;
		cout << argv[0] << " DATE TIME" << endl;
		cout << "where DATE is in year-mm-dd format, and TIME is in hh:mm:ss format." << endl;
		cout << "output is the time_t value." << endl;
		return 0;
	}
	char* date = argv[1];
	char* time = argv[2];

	struct tm timeinfo;
				int year, month, day, hour, min, sec;
	sscanf (date,"%d-%d-%d",&year, &month, &day);
	sscanf (time,"%d:%d:%d",&hour, &min, &sec);

	memset(&timeinfo, 0, sizeof(struct tm));
	timeinfo.tm_year = year - 1900; // year is "years since 1900"
	timeinfo.tm_mon = month - 1 ;   // month is in range 0...11
	timeinfo.tm_mday = day;         // range 1...31
	timeinfo.tm_hour = hour;        // range 0...23
	timeinfo.tm_min = min;          // range 0...59
	timeinfo.tm_sec = sec;          // range 0.
	
	time_t latest = timegm(&timeinfo);
	if(latest == -1) { cout << "Can not parse time date" << endl; exit(-1);}
	
	cout << (long)latest << endl;
	
	return 0;
}
