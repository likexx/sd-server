def validate(addr):
    if addr.count("::") > 1:
        return False
    # validChars = ['0','1','2','3',]
    for c in addr:
        if c.isdigit() or c in ['a','b','c','d','e','f',':','.']:
            continue
        return False
    
    return True

def hexTobyte(hex):
    value = 0
    for c in hex:
        if c>='0' and c<='9':
            value = value*16+int(c)
        else:
            value = value*16 + 10 + ord(c)-ord('a')
    return value

def addrToBytes(addr):
    while len(addr)<4:
        addr = "0"+addr
    byte1 = hexTobyte(addr[:2])
    byte2 = hexTobyte(addr[2:4])
    return [byte1, byte2]

def parseIpv4(addr):
    values = addr.split('.')
    if len(values)!=4:
        return None
    bytes = []
    for v in values:
        bytes.append(int(v))
    return bytes

def parseIpv6(addr):
    addr = addr.lower()
    if not validate(addr):
        return []
    # treat for normal case
    splitted = addr.split(":")
    values = []
    skipEmpty = False
    for v in splitted:
        if len(v)>0:
            values.append(v)
        else:
            if not skipEmpty:
                values.append(v)
                skipEmpty = True

    result = []
    if len(values)==8:
        for v in values:
            result += addrToBytes(v)
        return result

    # check if ipv4 exists: 1203:405::a0b:0:0.0.255.254
    hasIpv4 = False
    lastAddr = values[-1]
    ipv4Addr = []
    if lastAddr.find('.')>=0:
        ipv4Addr = parseIpv4(lastAddr)
        if ipv4Addr:
            hasIpv4 = True
    
    padded = []
    for v in values:
        if len(v)!=0:
            padded.append(v)
        else:
            expectLen = 8
            if hasIpv4:
                expectLen -= 1
            paddingNumber = expectLen - len(values) + 1
            for _ in range(paddingNumber):
                padded.append("0")

    if hasIpv4:
        padded = padded[:-1]

    for v in padded:
        result += addrToBytes(v)
    if hasIpv4:
        result = result + ipv4Addr
    
    if len(result)!=16:
        return []

    for v in result:
        if v == None or v<0 or v>255:
            return []

    return result
    

# print(addrToBytes("0"))
# print(parseIpv6("1234:abcd:0000:0000:0102:0000:0000:fffe"))
# print(parseIpv4("0.0.255.254"))
# print(parseIpv4("192.1.255.254"))

for test in ["1234:abcd:0000:0000:0102:0000:0000:fffe", 
             "1234:abcd:0:0:102:0:0:fffe", 
             "1234:ABCD::102:0:0:fffe", 
             "::ffff:c0a8:1", 
             "::",
             "::1",
             "123:456::",
             "::c080:1",
             "::192.128.0.1",
             "1234",
             "ab::c::",
             "ab   :"]:
    print(parseIpv6(test))
