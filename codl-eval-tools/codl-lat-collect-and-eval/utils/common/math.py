
def RoundUp(i, f):
  count = i // f
  return (count + 1) * f if i % f > 0 else i

def RoundUpV2(n, b):
  return ((n + b - 1) // b) * b

def RoundUpMul(a, b, f):
  #print('a {} b {} f {}'.format(a, b, f))
  i = a * b
  if f > 0:
    count = 0
    while i >= f:
      i = i - f
      count = count + 1
    if i > 0:
      count = count + 1
    return count * f
  else:
    return i

def RoundUpDiv(i, f):
  return (i + f - 1) // f

def LeastCommonMultiple(x, y):
  # choose the greater number
  if x > y:
    greater = x
  else:
    greater = y

  while(True):
    if((greater % x == 0) and (greater % y == 0)):
      lcm = greater
      break
    greater += 1

  return lcm // x, lcm
