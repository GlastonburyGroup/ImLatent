# Inspired by: Nicolas Flandrois [https://raw.githubusercontent.com/NicolasFlandrois/Stardate/master/sdcompute.py]

import datetime
import json

reference = {"earthdate": {"y": 2265, "m": 4, "d": 25}, "stardate": 0000.0}  # Dating refernce: Star Trek TOS
# reference = {"earthdate": {"y": 2256, "m": 5, "d": 11}, "stardate": 1207.3}  # Dating refernce: Star Trek Discovery Pilot, following Star Trek TOS
# reference = {"earthdate": {"y": 2364, "m": 5, "d": 11}, "stardate": 41986.0}  # Dating refernce: Star Trek TNG: Neutral Zone

class ComputeSD(object):
    """
    The Compute class regroup all necessary computing functions.
    """

    @staticmethod
    def leapyr(year: int):
        """"
        This function defines if the year is
        a Leap year (366 days)
        or a Normal year (365 days).
        Then it will to the variable n the value of 366 or 365, accordingly.
        """
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            n = 366
            # print("The year is a Leap year.\n")

        else:
            n = 365
            # print("The year is a normal year.\n")

        return n

    @staticmethod
    def nowearthdate():
        """Will generate automaticaly a tuple datetime object, for now time"""
        nowdate = datetime.datetime.now()
        return nowdate.timetuple(), nowdate.strftime('%A, %Y %B %d. %H:%M:%S')
    
    @staticmethod
    def nowstardate():
        """Will generate automaticaly the current stardate"""
        nowdate, _ = ComputeSD.nowearthdate()
        return ComputeSD.sdconvert(nowdate)

    @staticmethod
    def sdconvert(t):
        """
        Stardate calculator
        t = Time  (cf 'datetime.datetime.now().timetuple()' format)
        reference["earthdate"] = Earthdate Year reference point
        reference["stardate"] = Stardate Yaer reference point
        Compute.leapyr(t.tm_year) = number of days leap year/not (365 or 366)
        """
        return float(format(((reference["stardate"] +
                        (1000*(t.tm_year - reference["earthdate"]["y"]))) +
                      ((1000/((ComputeSD.leapyr(t.tm_year))*1440.0))*(((
                            t.tm_yday - 1.0)*1440.0) +
                        (t.tm_hour*60.0) + t.tm_min))), '.2f'))

    @staticmethod
    def sdtranslate(sd):
        """
        Stardate translator
        sd = Stardate Time  (cf float, stardate format)
        reference["earthdate"] = Earthdate Year reference point
        reference["stardate"] = Stardate Yaer reference point
        Compute.leapyr(t.tm_year) = number of days leap year/not (365 or 366)
        """
        dlist = []
        ed_year = int(((sd - reference["stardate"]) // 1000) +
                      reference["earthdate"]['y'])
        dlist.append(int(ed_year))
        ed_time = (((sd - reference["stardate"]) % 1000) /
                   (1000 / (1440*ComputeSD.leapyr(ed_year))))
        ed_day = (ed_time//1440)+1
        dlist.append(int(ed_day))
        ed_hour = (ed_time-((ed_day-1)*1440))//60
        dlist.append(int(ed_hour))
        ed_min = ed_time % 60
        dlist.append(int(ed_min))
        # NOTE: This calculation has 2 min leap from real date
        dstring = " ".join([str(i) for i in dlist])

        return datetime.datetime.strptime(dstring, '%Y %j %H %M').strftime(
            '%A, %d/%m/%Y %H:%M:%S')
    

# sd = ComputeSD.nowstardate()
# print(sd)
# print(ComputeSD.sdtranslate(float(sd)))