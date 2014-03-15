from obspy.fdsn.header import DEFAULT_USER_AGENT, \
    URL_MAPPINGS, DEFAULT_PARAMETERS, PARAMETER_ALIASES, \
    WADL_PARAMETERS_NOT_TO_BE_PARSED, FDSNException, FDSNWS
import urllib
import urllib2
    
def get_dataless_seed(network, station, location, channel, starttime,endtime):
    locs = locals()  
    #print locs
    param = {}
    param['net'] = network
    param['sta'] = station
    param['loc'] = location
    param['cha'] = channel
    param['start'] = starttime.strftime("%Y-%m-%dT%H:%M:%S")
    param['end'] = endtime.strftime("%Y-%m-%dT%H:%M:%S")
    base_url ='http://service.ncedc.org'
    service = 'dataless'
    url = build_url(base_url,service,1,'query',param)
    print url
    return url
    
def build_url(base_url, service, major_version, resource_type, parameters={}):
    """
    URL builder for the FDSN webservices.

    Built as a separate function to enhance testability.

    >>> build_url("http://service.iris.edu", "dataselect", 1, \
                  "application.wadl")
    'http://service.iris.edu/fdsnws/dataselect/1/application.wadl'

    >>> build_url("http://service.iris.edu", "dataselect", 1, \
                  "query", {"cha": "EHE"})
    'http://service.iris.edu/fdsnws/dataselect/1/query?cha=EHE'
    """
    # Only allow certain resource types.
    if service not in ["dataless"]:
        msg = "Resource type '%s' not allowed. Allowed types: \n%s" % \
            (service, ",".join(("dataless")))
        raise ValueError(msg)

    # Special location handling.
    if "location" in parameters:
        loc = parameters["location"].replace(" ", "")
        # Empty location.
        if not loc:
            loc = "--"
        # Empty location at start of list.
        if loc.startswith(','):
            loc = "--" + loc
        # Empty location at end of list.
        if loc.endswith(','):
            loc += "--"
        # Empty location in middle of list.
        loc = loc.replace(",,", ",--,")
        parameters["location"] = loc

    url = "/".join((base_url, "ncedcws", service,
                    str(major_version), resource_type))
    if parameters:
        # Strip parameters.
        for key, value in parameters.iteritems():
            try:
                parameters[key] = value.strip()
                #print key, parameters[key]
            except:
                pass
        url = "?".join((url, urllib.urlencode(parameters)))
    return url