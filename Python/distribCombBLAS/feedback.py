import os
import IPython.ipapi
import pwd
import email
import smtplib
import socket
#import IPython.Magic as Magic

_kdt_LogFname = '.KDT_log';
_kdt_EmailFname = 'KDT_email';
_kdt_Alias = 'steve.reinhardt@microsoft.com';
_kdt_Nlines2Send = 30;


def startFeedback():
	ip = IPython.ipapi.get()
	ip.magic('logstart %s over' % _kdt_LogFname);


def sendFeedback(nlines=_kdt_Nlines2Send):
	if not os.path.exists(_kdt_LogFname) or not os.path.isfile(_kdt_LogFname):
		print "logging apparently not enabled for KDT"
		#return;
	
	logFid = open(_kdt_LogFname);
	count = 0;
	# count the lines in the file
	while 1:
		line = logFid.readline()
		if not line:
			break;
		count += 1;
	logFid.close();
	logFid = open(_kdt_LogFname);
	# space past all but last N lines
	for i in range(count - nlines):   
		logFid.readline();
	# copy last N lines to new file
	emailFid = open(_kdt_EmailFname,'w+');
	for i in range(nlines):
		line = logFid.readline();
		emailFid.writelines(line);
	logFid.close();
	emailFid.close();
		
	str = "The code example you want to send to the KDT developers is \nin %s/%s.  \nIf you wish, edit it with your favorite editor.  Type 'Go' \nwhen you are ready to send it or 'Cancel' to cancel sending it.\n>>>" % (os.getcwd(),_kdt_EmailFname)
		
	resp = raw_input(str);
	if resp == 'Go' or resp == 'go':
		#print "Emailing the file."
		msg = email.message_from_file(open(_kdt_EmailFname));
		pwdEntry = pwd.getpwuid(os.getuid());
		userAddress = '%s@%s' % (pwdEntry[0], socket.gethostname());
		userFullName = pwdEntry[4];
		msg['Subject'] = 'KDT feedback from %s' % userFullName;
		msg['From'] = userAddress;
		msg['To'] = _kdt_Alias;
		s = smtplib.SMTP('localhost');
		s.sendmail(userAddress, [_kdt_Alias], msg.as_string());
		s.quit();
	else:
		print "Canceling the send."


#startFeedback();
