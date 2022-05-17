
class TimeUtils(object):
  @staticmethod
  def _format1(hour, minute, second):
    text = ''
    if hour != -1:
      text = text + str(hour) + ' h '
    if minute != -1:
      text = text + str(minute) + ' m '
    text = text + str(second) + ' s'
    return text

  @staticmethod
  def _format2(hour, minute, second):
    hour = 0 if hour == -1 else hour
    minute = 0 if minute == -1 else minute
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hour, minute, second)

  @staticmethod
  def sec_to_format_text(seconds):
    hour = -1
    minute = -1
    if seconds >= 60 * 60:
      hour = seconds // (60 * 60)
      seconds = seconds - hour * 60 * 60
    if seconds >= 60:
      minute = seconds // 60
      seconds = seconds - minute * 60

    return TimeUtils._format2(hour, minute, seconds)

if __name__ == '__main__':
  print(TimeUtils.sec_to_format_text(1))
  print(TimeUtils.sec_to_format_text(60))
  print(TimeUtils.sec_to_format_text(61))
  print(TimeUtils.sec_to_format_text(3600))
  print(TimeUtils.sec_to_format_text(3660))
  print(TimeUtils.sec_to_format_text(3661))
