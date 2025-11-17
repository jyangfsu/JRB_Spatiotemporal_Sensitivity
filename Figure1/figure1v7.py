#!/usr/bin/env python

# Copyright 2019-2020 Juliane Mai - juliane.mai(at)uwaterloo.ca
#
# License
# This file is part of Juliane Mai's personal code library.
#
# Juliane Mai's personal code library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Juliane Mai's personal code library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A  PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public Licensefstop
# along with Juliane Mai's personal code library.  If not, see <http://www.gnu.org/licenses/>.
#
# run with:
#     run figure_2.py -t pdf -p figure_2

from __future__ import print_function

"""

Plots Salmon River catchment and annual precipitation and temperature

History
-------
Written,  JM, Jul 2019
"""

# -------------------------------------------------------------------------
# Command line arguments - if script
#

# Comment|Uncomment - Begin
if __name__ == '__main__':

    import argparse
    import numpy as np

    plotname    = ''
    outtype     = ''
    usetex      = True
    serif       = False
    doabc       = True
    nsets       = 10            # number of Sobol sequences
    nboot       = 1             # Set to 1 for single run of SI and STI calculation
    
    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Benchmark example to test Sensitivity Analysis for models with multiple process options.''')
    parser.add_argument('-p', '--plotname', action='store',
                        default=plotname, dest='plotname', metavar='plotname',
                        help='Name of plot output file for types pdf, html or d3, '
                        'and name basis for type png (default: '+__file__[0:__file__.rfind(".")]+').')
    parser.add_argument('-s', '--serif', action='store_true', default=serif, dest="serif",
                    help="Use serif font; default sans serif.")
    parser.add_argument('-t', '--type', action='store',
                        default=outtype, dest='outtype', metavar='outtype',
                        help='Output type is pdf, png, html, or d3 (default: open screen windows).')
    parser.add_argument('-u', '--usetex', action='store_true', default=usetex, dest="usetex",
                        help="Use LaTeX to render text in pdf, png and html.")
    parser.add_argument('-b', '--nboot', action='store',
                        default=nboot, dest='nboot', metavar='nboot',
                        help='Number of bootstrap samples (default: nboot=10).')
    parser.add_argument('-n', '--nsets', action='store',
                        default=nsets, dest='nsets', metavar='nsets',
                        help='Number of sensitivity samples (default: nsets=10).')
    

    args     = parser.parse_args()
    plotname = args.plotname
    outtype  = args.outtype
    serif    = args.serif
    usetex   = args.usetex
    nboot    = np.int32(args.nboot)
    nsets    = np.int32(args.nsets)
    
    


    del parser, args
    # Comment|Uncomment - End

    # -----------------------
    # add subolder scripts/lib to search path
    # -----------------------
    import sys
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path + '/../lib')


# -------------------------------------------------------------------------
# Function definition - if function
#

    # Check input
    outtype = outtype.lower()
    outtypes = ['', 'pdf', 'png', 'html', 'd3']
    if outtype not in outtypes:
        print('\nError: output type must be in ', outtypes)
        import sys
        sys.exit()

    import numpy as np
    import xarray as xr
    import time
    import datetime
    import color                           # in lib/
    from   position        import position # in lib/
    from   autostring      import astr     # in lib/
    from   abc2plot        import abc2plot # in lib/
    from   fread           import fread    # in lib/
    from   str2tex         import str2tex  # in lib/

    
    t1 = time.time()

    if (outtype == 'd3'):
        try:
            import mpld3
        except:
            print("No mpld3 found. Use output type html instead of d3.")
            outtype = 'html'

    

    # -------------------------------------------------------------------------
    # Setup
    #
    dowhite    = False  # True: black background, False: white background
    title      = False   # True: title on plots, False: no plot titles
    textbox    = False  # if true: additional information is set as text box within plot
    textbox_x  = 0.95
    textbox_y  = 0.85

    # -------------------------------------------------------------------------
    # Setup Calculations
    #
    if dowhite:
        fgcolor = 'white'
        bgcolor = 'black'
    else:
        fgcolor = 'black'
        bgcolor = 'white'

    # colors
    cols1 = color.get_brewer('YlOrRd9', rgb=True)
    cols1 = color.get_brewer( 'WhiteYellowOrangeRed',rgb=True)[30:]
    cols1 = color.get_brewer( 'dark_rainbow_256',rgb=True)   # blue to red

    cols2 = color.get_brewer('YlOrRd9', rgb=True)[::-1]
    cols2 = color.get_brewer( 'WhiteYellowOrangeRed',rgb=True)[30:][::-1]
    cols2 = color.get_brewer( 'dark_rainbow_256',rgb=True)[::-1]  # red to blue

    cols3 = [cols2[0],cols2[95],cols2[-1]]  # red, yellow, blue
    cols3 = [color.colours('gray'),cols2[0],color.colours('white')]  # gray red white

    # -------------------------------------------------------------------------
    # Read latlon info from file
    # -------------------------------------------------------------------------
    latlon = fread("Jinghe_latlon.dat",skip=1)

    # -------------------------------------------------------------------------
    # Read forcings from file
    # -------------------------------------------------------------------------
    forcing = fread("obs.rvt", skip=4, comment=":")
    ntime     = np.shape(forcing)[0]
    ntime_doy = 365
    forc_time = np.array([ datetime.datetime(1954,1,1,0,0) + datetime.timedelta(itime) for itime in range(ntime) ])
    leap      = np.array([ True if (forc_time[ii].month == 2 and forc_time[ii].day == 29) else False for ii in range(ntime) ])

    # RAINFALL   SNOWFALL   TEMP_DAILY_MIN   TEMP_DAILY_MAX   PET
    rain      = forcing[~leap,0]
    snow      = forcing[~leap,1]
    tmin      = forcing[~leap,2]
    tmax      = forcing[~leap,3]
    pet       = forcing[~leap,4]
    pre       = forcing[~leap,0]+forcing[~leap,1]
    tavg      = (tmin+tmax)/2.0

    # average over day of year
    forc_time = forc_time[~leap]
    pre_doy  = np.median(np.reshape(pre[:],  [int(ntime/ntime_doy),ntime_doy]),axis=0)
    snow_doy = np.median(np.reshape(snow[:], [int(ntime/ntime_doy),ntime_doy]),axis=0)
    rain_doy = np.median(np.reshape(rain[:], [int(ntime/ntime_doy),ntime_doy]),axis=0)
    tavg_doy = np.mean(np.reshape(tavg[:],   [int(ntime/ntime_doy),ntime_doy]),axis=0)

    # mean annual precip
    years = [ tt.year for tt in forc_time ]
    uyears = np.unique(years)
    annual_precip = []
    for uu in uyears:
        idx = np.where(years == uu)
        annual_precip.append(np.sum(pre[idx]))
    annual_temp = np.mean(tavg_doy)
    # print("annual_precip      = ", annual_precip)
    #print()
    #print('---------------------------------------------')
    #print("mean annual temp   = ", np.mean(annual_temp))
    #print("mean annual precip = ", np.mean(annual_precip))
    #print('---------------------------------------------')
    #print()

    # mean monthly temperature and average total precip per month
    years  = [ tt.year for tt in forc_time ]
    months = [ tt.month for tt in forc_time ]
    uyears = np.unique(years)
    month_prec = []
    month_snow = []
    month_rain = []
    month_temp = []
    for uu in uyears:
        tmp_prec = []
        tmp_snow = []
        tmp_rain = []
        tmp_temp = []
        for mm in np.arange(1,13):
            idx = np.where((years == uu) & (months == mm))
            tmp_prec.append(np.sum(pre[idx]))
            tmp_snow.append(np.sum(snow[idx]))
            tmp_rain.append(np.sum(rain[idx]))
            tmp_temp.append(np.mean(tavg[idx]))
        month_prec.append(tmp_prec)
        month_snow.append(tmp_snow)
        month_rain.append(tmp_rain)
        month_temp.append(tmp_temp)
    month_prec = np.mean(np.array(month_prec),axis=0)
    month_snow = np.mean(np.array(month_snow),axis=0)
    month_rain = np.mean(np.array(month_rain),axis=0)
    month_temp = np.mean(np.array(month_temp),axis=0)

    # -------------------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------------------
    ocean_color = (151/256., 183/256., 224/256.)
    infil_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[20]
    quick_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[55]
    evapo_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[80]
    basef_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[105]
    snowb_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[130]
    convs_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[155]
    convd_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[180]
    potme_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[205]
    perco_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[230]
    soilm_color = color.get_brewer( 'WhiteBlueGreenYellowRed',rgb=True)[255]
    
    # -------------------------------------------------------------------------
    # Plotting of results
    # -------------------------------------------------------------------------
    # Main plot
    ncol        = 2           # number columns
    nrow        = 4           # number of rows
    textsize    = 9         # standard text size
    dxabc       = 0.03          # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
    dyabc       = 0.92          # % of (max-min) shift up from lower x-axis for a,b,c,... labels
    dxsig       = 1.23        # % of (max-min) shift to the right from left y-axis for signature
    dysig       = -0.075      # % of (max-min) shift up from lower x-axis for signature
    dxtit       = 0           # % of (max-min) shift to the right from left y-axis for title
    dytit       = 1.2         # % of (max-min) shift up from lower x-axis for title
    hspace      = 0.16        # x-space between subplots
    vspace      = 0.06        # y-space between subplots

    lwidth      = 0.5         # linewidth
    elwidth     = 0.5         # errorbar line width
    alwidth     = 1.0         # axis line width
    glwidth     = 0.5         # grid line width
    msize       = 8.0         # marker size
    mwidth      = 0.0         # marker edge width
    mcol1       = '0.7'       # primary marker colour
    mcol2       = '0.0'       # secondary
    mcol3       = '0.0'       # third
    mcols       = color.colours(['blue','green','yellow','orange','red','darkgray','darkblue','black','darkgreen','gray'])
    lcol0       = color.colours('black')    # line colour
    lcol1       = color.colours('blue')     # line colour
    lcol2       = color.colours('green')    # line colour
    lcol3       = color.colours('yellow')   # line colour
    lcols       = color.colours(['black','blue','green','yellow'])
    markers     = ['o','v','s','^']

    # Legend
    llxbbox     = 0.98        # x-anchor legend bounding box
    llybbox     = 0.98        # y-anchor legend bounding box
    llrspace    = 0.          # spacing between rows in legend
    llcspace    = 1.0         # spacing between columns in legend
    llhtextpad  = 0.4         # the pad between the legend handle and text
    llhlength   = 1.5         # the length of the legend handles
    frameon     = False       # if True, draw a frame around the legend. If None, use rc
      
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle, Circle, Polygon
    from mpl_toolkits.basemap import Basemap
    mpl.use('pdf')
    
    if (outtype == 'pdf'):
        mpl.use('PDF') # set directly after import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        # Customize: http://matplotlib.sourceforge.net/users/customizing.html
        mpl.rc('ps', papersize='a4', usedistiller='xpdf') # ps2pdf
        # mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
        if usetex:
            mpl.rc('text', usetex=True)
            if not serif:
                #   r'\usepackage{helvet}',                             # use Helvetica
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage[math,lf,mathtabular,footnotefigures]{MyriadPro}', # use MyriadPro font
                    r'\renewcommand{\familydefault}{\sfdefault}',       # normal text font is sans serif
                    r'\figureversion{lining,tabular}',
                    r'\usepackage{wasysym}',                            # for permil symbol (load after MyriadPro)
                    ]
            else:
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage{wasysym}'                     # for permil symbol
                    ]
        else:
            if serif:
                mpl.rcParams['font.family']     = 'serif'
                mpl.rcParams['font.sans-serif'] = 'Times'
            else:
                mpl.rcParams['font.family']     = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = 'Arial'       # Arial, Verdana
    elif (outtype == 'png') or (outtype == 'html') or (outtype == 'd3'):
        mpl.use('Agg') # set directly after import matplotlib
        import matplotlib.pyplot as plt
        # mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
        if usetex:
            mpl.rc('text', usetex=True)
            if not serif:
                #   r'\usepackage{helvet}',                             # use Helvetica
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage[math,lf,mathtabular,footnotefigures]{MyriadPro}', # use MyriadPro font
                    r'\renewcommand{\familydefault}{\sfdefault}',       # normal text font is sans serif
                    r'\figureversion{lining,tabular}',
                    r'\usepackage{wasysym}',                            # for permil symbol (load after MyriadPro)
                    ]
            else:
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage{wasysym}'                     # for permil symbol
                    ]
        else:
            if serif:
                mpl.rcParams['font.family']     = 'serif'
                mpl.rcParams['font.sans-serif'] = 'Times'
            else:
                mpl.rcParams['font.family']     = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = 'Arial'       # Arial, Verdana
        mpl.rc('savefig', dpi=dpi, format='png')
    else:
        import matplotlib.pyplot as plt
        # mpl.rc('figure', figsize=(4./5.*8.27,4./5.*11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
        
    mpl.rc('text.latex') #, unicode=True)
    mpl.rc('font', size=textsize)
    mpl.rc('path', simplify=False) # do not remove
    # print(mpl.rcParams)
    mpl.rc('axes', linewidth=alwidth, edgecolor=fgcolor, facecolor=bgcolor, labelcolor=fgcolor)
    mpl.rc('figure', edgecolor=bgcolor, facecolor='grey')
    mpl.rc('grid', color=fgcolor)
    mpl.rc('lines', linewidth=lwidth, color=fgcolor)
    mpl.rc('patch', edgecolor=fgcolor)
    mpl.rc('savefig', edgecolor=bgcolor, facecolor=bgcolor)
    mpl.rc('patch', edgecolor=fgcolor)
    mpl.rc('text', color=fgcolor)
    mpl.rc('xtick', color=fgcolor)
    mpl.rc('ytick', color=fgcolor)
    
    import matplotlib
    # ---------- WRR-like minimalist defaults ----------
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']   # fallback if missing
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype']  = 42
    
    
    
    plt.rcParams.update({'hatch.color': 'grey'})
    
    plt.rcParams.update({
   "text.usetex": True,
   "font.family": "sans-serif"})
    
    """Reasonable WRR-like defaults."""
    mpl.rcParams.update({
        # Core sizing
        "figure.dpi": 100,
        "savefig.dpi": 300,

        # Fonts
        "font.size": 9.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"],
        "mathtext.default": "regular",
        "mathtext.fontset": "dejavusans",   # matches sans-serif

        # Embed fonts properly in PDF/PS
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Axes & ticks
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,

        # Lines & markers
        "lines.linewidth": 0.8,
        "lines.markersize": 4,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 8.0,
        "legend.handlelength": 1.2,

        # Grid off by default
        "axes.grid": False,

        # Savefig background
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.constrained_layout.use": True,   # cleaner spacing
    })


    if (outtype == 'pdf'):
        pdffile = plotname+'.pdf'
        print('Plot PDF ', pdffile)
        pdf_pages = PdfPages(pdffile)
    elif (outtype == 'png'):
        print('Plot PNG ', plotname)
    else:
        print('Plot X')

    t1  = time.time()
    ifig = 0


    usetex = True
    
    ifig = 0
    
    ifig += 1
    iplot = 0
    print('Plot - Fig ', ifig)
    fig = plt.figure(ifig)

    # -----------------------
    # plot
    # -----------------------
    ylim = [-0.1, 0.6]

    # -------------
    # Salmon River watershed
    # -------------
    iplot += 1
    #                           [left, bottom, width, height]
    sub = fig.add_axes(np.array([ 0.0,  0.745 ,  0.3475,  0.155 ]))

    # Map: Canada - Lake Erie
    llcrnrlon =   102.0 #-81.25 #-85.5
    urcrnrlon =   127.0 #-77.0
    llcrnrlat =   26.0 #39.5
    urcrnrlat =   44.0
    lat_1     =   37.0 # 42.0  # first  "equator"
    lat_2     =   37.0 # 42.0  # second "equator"
    lat_0     =   37.0 # 42.5  # center of the map
    lon_0     =   118.5 #-79.00 #-82.0  # center of the map
    # m = Basemap(projection='lcc',
    #             llcrnrlon=-80, llcrnrlat=43, urcrnrlon=-75, urcrnrlat=47,
    #             lon_0=-77.5, lat_0=43, 
    #             lat_1=44, lat_2=44, 
    #             resolution='i') # Lambert conformal
    m = Basemap(projection='lcc', area_thresh=10000.,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
                resolution=None) # Lambert conformal

    # draw parallels and meridians.
    # labels: [left, right, top, bottom]
    if iplot%ncol == 1:  # plot in 1st column
        parallels = m.drawparallels(np.arange(-80.,81.,5.),  labels=[1,0,0,0], dashes=[1,1], linewidth=0.25, color='0.5', )
    elif iplot%ncol == 0:  # plot in last column
        parallels = m.drawparallels(np.arange(-80.,81.,5.),  labels=[0,1,0,0], dashes=[1,1], linewidth=0.25, color='0.5')
    else:
        parallels = m.drawparallels(np.arange(-80.,81.,5.),  labels=[0,0,0,0], dashes=[1,1], linewidth=0.25, color='0.5')
    
   
    meridians = m.drawmeridians(np.arange(-175.,181.,10.),labels=[0,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
    
    # === Make latitude & longitude labels italic ===
    for par_dict in [parallels, meridians]:
        for key in par_dict:
            for text in par_dict[key][1]:  # [1] -> list of Text objects
                text.set_fontstyle('normal')  # italic font
                text.set_fontsize(9)          # optional: set font size
                text.set_color('black')       # optional: set color
                
                
    for key, (line, texts) in parallels.items():
        for text in texts:
            lat = key
            if lat > 0:
                new_label = rf"${abs(lat):.0f}^\circ\mathrm{{N}}$"
            else:
                new_label = rf"${abs(lat):.0f}^\circ\mathrm{{S}}$"
            text.set_text(str2tex(new_label, usetex=True))             
            text.set_color('black')

    for key, (line, texts) in meridians.items():
        for text in texts:
            lon = key
            if lon > 0:
                new_label = rf"${abs(lon):.0f}^\circ\mathrm{{E}}$"
            else:
                new_label = rf"${abs(lon):.0f}^\circ\mathrm{{W}}$"
            text.set_text(str2tex(new_label, usetex=True))  
            text.set_color('black')
        
    # draw cooastlines and countries
    #m.drawcoastlines(linewidth=0.3)
    #m.drawmapboundary(fill_color=ocean_color, linewidth=0.3)
    #m.drawcountries(color='black', linewidth=0.3)
    # m.fillcontinents(lake_color=ocean_color)  # color='white', 
    m.shadedrelief()
    # m.drawlsmask()

    # convert coordinates to map coordinates
    xpt,ypt = m(latlon[:,1],latlon[:,0])
    mlatlon = np.transpose(np.array([xpt,ypt])) #zip(xpt,ypt)

    # Catchments
    facecolor = perco_color
    sub.add_patch(Polygon(mlatlon, facecolor=facecolor, edgecolor='none', alpha=1.0))

    # some reference points
    nref = 1
    ref_latlon = np.ones([nref,2]) * -9999.0
    ref_names  = []
    ref_latlon[0,0] = 39.9333 ; ref_latlon[0,1] =  116.3333  ; ref_names.append( 'Beijing' )   
    #ref_latlon[1,0] = 34.2658 ; ref_latlon[1,1] =  108.9541  ; ref_names.append( 'Xi\'an' )   
    #ref_latlon[2,0] = 35.2057 ; ref_latlon[2,1] =  107.7987  ; ref_names.append( 'Changwu' )   
   

    for iref in range(nref):
            xpt1, ypt1 = m(ref_latlon[iref,1],ref_latlon[iref,0])
            sub.plot(xpt1, ypt1,
                     #linewidth=lwidth,
                     #color=color,
                     marker='*',
                     markeredgecolor='black',
                     markerfacecolor='#DA2322',
                     markersize=6, #msize,
                     markeredgewidth=mwidth)
            ha = ['right', 'left']
            va = ['bottom','top']

            if iref in [0, 2]:
                ha = 'left'
                dx = +0.75
            else:
                ha = 'left'
                dx = -0.3
            x2, y2 = m(ref_latlon[iref,1]+dx,ref_latlon[iref,0])
            
            # Enable LaTeX rendering for text
            plt.rc('text', usetex=usetex)
            plt.rc('font', family='Helvetica')
            sub.text(x2, y2, str2tex(ref_names[iref], usetex=usetex), fontsize=textsize)
            
            '''
            sub.annotate(ref_names[iref],
                xy=(xpt1, ypt1),  xycoords='data',
                xytext=(x2, y2), textcoords='data',
                fontsize='small',
                verticalalignment='center',horizontalalignment=ha#,
                # arrowprops=dict(arrowstyle="->",relpos=(1.0,0.5),linewidth=0.4)
                )
            '''

    # Title
    #sub.set_title(str2tex('Salmon River watershed',usetex=usetex),fontsize=textsize)

    # Fake subplot for numbering
    if doabc:
        pos = position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)
        lsub = fig.add_axes([0.245, 0.88, 0.15, 0.0155])

        lsub.set_xlim([0,1])
        lsub.set_ylim([0,1])

        # subplot numbering
        abc2plot(lsub, dxabc, dyabc, iplot, lower=True, parenthesis='both',
                     bold=True, large=True,
                     mathrm=True, usetex=usetex,
                     horizontalalignment='left', verticalalignment='top')

        lsub.set_title('')
        lsub.set_xlabel('')
        lsub.set_ylabel('')
        lsub.set_xticks([])
        lsub.set_yticks([])
        lsub.set_axis_off()

    # -------------
    # Annual precipitation and temperature
    # -------------
    #month_snow == month ET here

    #month_snow == month ET here
    # From SWAT 1972-1986  "output_1962_1986.csv" by calculate_mean_monthly_PCP_ET
    month_snow = np.array([26.590, 39.586, 81.067978, 106.505, 124.381, 129.809, 120.077, 114.154, 79.922, 71.256, 49.069, 29.585])
    
    #month_pcec = np.array([4.21, 6.20, 15.69, 30.13, 48.25, 55.25, 98.46, 101.95, 80.97, 52.50, 14.25, 3.03])
    # From SWAT 1972-1986
    month_pcec = np.array([4.557, 6.744, 19.446, 43.052, 61.937, 59.54, 122.783, 117.259, 116.034, 49.572, 20.134, 4.452])
    month_smelt = np.array([3.930, 7.738, 9.063, 1.053, 0, 0, 0, 0, 0, 0.8769, 2.996, 3.127])
    
    month_rain = month_pcec + month_smelt 
    
    month_temp = np.array([-5.266, -2.693, 3.789, 10.323, 15.591, 19.704, 21.633, 20.504, 14.807, 9.388, 2.13, -4.025])
     
    iplot += 1
    #                           [left, bottom, width, height]
    sub = fig.add_axes(np.array([ 0.4125,  0.745 ,  0.4475,  0.155 ]))

    # make color a bit darker for reviewer (use color of "water" in panels C and D)
    quick_color_dark = '#508ABD'

    #ppre   = sub.plot( np.arange(ntime_doy), pre_doy,  color = quick_color)
    #psnow  = sub.plot( np.arange(ntime_doy), snow_doy,  color = infil_color)
    #prain  = sub.plot( np.arange(ntime_doy), rain_doy,  color = quick_color)
    width = 0.85
    #psnow  = sub.bar( np.arange(12), month_snow, width, color = quick_color_dark, hatch='xxxxx')
    #prain  = sub.bar( np.arange(12), month_rain, width, color = quick_color, alpha=0.75)

    rain_bar = sub.bar(np.arange(12), month_rain, width=width, alpha=0.85, label="Precipitation")
    et_bar   = sub.bar(np.arange(12), month_snow, width=width, alpha=0.85, label="Evaporation",
                       facecolor="none", edgecolor="black", linewidth=0.6, hatch="////")
     
    sub2 = sub.twinx()
    # ptavg = sub2.plot( np.arange(ntime_doy), tavg_doy, color = perco_color)
    tline = sub2.plot( np.arange(12), month_temp, color = perco_color, linewidth = 2*lwidth)

    
    print('---------------------------------------------')
    print("mean annual temp   = ", np.mean(month_temp))
    print("mean annual precip = ", np.sum(month_pcec))
    print('---------------------------------------------')
 
 
    print("month_snow = ",month_snow)
    print("month_rain = ",month_rain)
    print("month_prec = ",month_prec)
    print("month_temp = ",month_temp)
    

    ylim = [0,150]
    sub.set_ylim(ylim)
    ylim = [-8, 30]
    sub2.set_ylim(ylim)

    # nopt = len(procopt)
    #sub.set_xticks(np.arange(12), ['J','F','M','A','M','J','J','A','S','O','N','D'])
    plt.rc('text', usetex=usetex)
    plt.rc('font', family='Helvetica')
    plt.xticks(np.arange(12), [str2tex(i, usetex=usetex) for i in ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']])
    
    
    # plt.title(str2tex('Sensitivities of Process Options',usetex=usetex))
    sub.set_xlabel(str2tex("Month",usetex=usetex), color='black')
    sub.set_ylabel(str2tex("Avg. Monthly Pcp./Evp. [mm]",usetex=usetex), color=quick_color_dark)
    #sub.set_yticks([0, 50, 100, 150], ['0', '50', '150', '150'], rotation=90)
    sub2.set_ylabel(str2tex('Avg. Monthly Temp. [$^\circ$C]',usetex=usetex), color=perco_color)
    #sub2.set_yticks([0, 10, 20, 30], ['0', '10', '20', '30'], rotation=90)
    
    # Unified legend (bars + line)
    #handles = [rain_bar, et_bar, tline]
    #labels  = ["Rain", "Evaporation", "Temperature"]
    #sub.legend(handles, labels, loc="upper left", frameon=False, ncol=1, handletextpad=0.6, columnspacing=1.0)
    
    
    # Create custom artists
    #      (left, bottom), width, height
    boxSTi_1 = patches.Rectangle(
    (0.045, 0.92), 0.02, 0.05,
    facecolor='white',        # fill is white
    edgecolor='black',        # hatch (and edge) will be black
    hatch='////',             # your hatch
    linewidth=0.6,            # thin WRR-style stroke
    alpha=1.0,
    fill=True,
    transform=sub.transAxes,
    clip_on=False
    )
    
    boxSTi_2  = patches.Rectangle( (0.045, 0.84), 0.02, 0.05, color = quick_color_dark, alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    line      = patches.Rectangle( (0.045, 0.78), 0.02, 0.00, color = perco_color,      alpha=1.0, fill  = True, transform=sub.transAxes, clip_on=False )
    sub.add_patch(boxSTi_1)
    sub.add_patch(boxSTi_2)
    sub.add_patch(line)
    sub.text(0.1, 0.94, str2tex("Evaporation", usetex=usetex),         fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.1, 0.86, str2tex("Precipitation",usetex=usetex),         fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    sub.text(0.1, 0.78, str2tex("Temperature",usetex=usetex),  fontsize='small', horizontalalignment='left', verticalalignment='center', transform=sub.transAxes)
    
    
    abc2plot(sub,0.93,0.96,iplot,bold=True, lower=True,
             usetex=usetex,mathrm=True, large=True, parenthesis='both',verticalalignment='top')

    # -------------
    # Salmon River watershed (soil)
    # -------------
    iplot += 1
    #                           [left, bottom, width, height]
    lleft   = -0.023
    bbottom = 0.435
    wwidth  = 0.5 #0.3475
    hheight = 0.25 #0.155
    sub = fig.add_axes(np.array([ lleft, bbottom, wwidth, hheight ]))

    # Map: Canada - Lake Erie
    llcrnrlon =  105.6 #-81.25 #-85.5
    urcrnrlon =  109.5 #-77.0
    llcrnrlat =   34.2 #39.5
    urcrnrlat =   37.6
    
    lat_1     =   (llcrnrlat+urcrnrlat)/2.0 # 42.0  # first  "equator"
    lat_2     =   (llcrnrlat+urcrnrlat)/2.0 # 42.0  # second "equator"
    lat_0     =   (llcrnrlat+urcrnrlat)/2.0 # 42.5  # center of the map
    lon_0     =   (llcrnrlon+urcrnrlon)/2.0 #-79.00 #-82.0  # center of the map
    # m = Basemap(projection='lcc',
    #             llcrnrlon=-80, llcrnrlat=43, urcrnrlon=-75, urcrnrlat=47,
    #             lon_0=-77.5, lat_0=43, 
    #             lat_1=44, lat_2=44, 
    #             resolution='i') # Lambert conformal
    m = Basemap(projection='lcc', #area_thresh=10000.,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
                resolution='i') # Lambert conformal
    

    # draw parallels and meridians.
    # labels: [left, right, top, bottom]
    parallels = m.drawparallels(np.arange(-80.,81., 1),  labels=[1,0,0,0], dashes=[1,1], linewidth=0.25, color='0.5')
    meridians = m.drawmeridians(np.arange(-180,181.1, 1),labels=[0,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
    
    
    # === Make latitude & longitude labels italic ===
    for par_dict in [parallels, meridians]:
        for key in par_dict:
            for text in par_dict[key][1]:  # [1] -> list of Text objects
               #text.set_fontstyle('italic')  # italic font
                text.set_fontsize(9)          # optional: set font size
                text.set_color('black')       # optional: set color

    # draw cooastlines and countries
    # m.drawcoastlines(linewidth=0.3)
    m.drawmapboundary(fill_color=ocean_color, linewidth=0.3)
    m.drawcountries(color='black', linewidth=0.3)
    m.fillcontinents(color='white', lake_color=ocean_color)

    # scalebar
    length = 100 # km
    m.drawmapscale(llcrnrlon*0.80+urcrnrlon*0.20, llcrnrlat*0.88+urcrnrlat*0.12, lon_0, lat_0, length, barstyle='fancy', fontsize='small')

    for text in sub.texts:
        if 'km' in text.get_text():   
            text.set_text(str2tex('km')) 
        if text.get_text() == '0':
            text.set_text(str2tex('0')) 
        if text.get_text() == '50':
            text.set_text(str2tex('50')) 
        if text.get_text() == '100':
            text.set_text(str2tex('100')) 
            
    for key, (line, texts) in parallels.items():
        for text in texts:
            lat = key
            if lat > 0:
                new_label = rf"${abs(lat):.0f}^\circ\mathrm{{N}}$"
            else:
                new_label = rf"${abs(lat):.0f}^\circ\mathrm{{S}}$"
            text.set_text(str2tex(new_label, usetex=True))             
            text.set_color('black')

    for key, (line, texts) in meridians.items():
        for text in texts:
            lon = key
            if lon > 0:
                new_label = rf"${abs(lon):.0f}^\circ\mathrm{{E}}$"
            else:
                new_label = rf"${abs(lon):.0f}^\circ\mathrm{{W}}$"
            text.set_text(str2tex(new_label, usetex=True))  
            text.set_color('black')




    # Load the NetCDF file
    from netCDF4 import Dataset
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    
    # --- Load DEM from NetCDF file ---
    file_path = 'DEM_WGS84_ProjectRaster.nc'
    nc = Dataset(file_path, 'r')
    
    # Read latitude, longitude, and elevation
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    z = nc.variables['dem'][:]  # Replace 'dem' if your variable name differs
    
    nc.close()
    
    # --- Ensure latitude is ascending ---
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        z = z[::-1, :]
    
    # Create meshgrid for lat and lon (2D arrays)
    lon, lat = np.meshgrid(lon, lat)
    
    # --- Compute hillshade using LightSource ---
    # LightSource simulates illumination from a given direction
    # azdeg = azimuth (light direction in degrees clockwise from north)
    # altdeg = altitude (height of the light above the horizon)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(z, cmap=plt.cm.terrain, vert_exag=10, blend_mode='overlay')
    
    # Convert RGB hillshade result to grayscale
    shade_gray = np.mean(rgb, axis=2)
    
    # Replace very dark pixels with NaN so they become transparent in overlay
    shade_gray[shade_gray <= 0.05] = np.nan
    
    # Normalize DEM values for color mapping
    z_norm = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))
    
    # Prepare terrain colormap
    cmap = plt.cm.terrain
    terrain_rgb = cmap(z_norm)
    
    # Multiply terrain color by hillshade intensity (optional enhancement)
    hillshade = terrain_rgb.copy()
    for i in range(3):
        hillshade[..., i] = hillshade[..., i] * shade_gray
    
    # --- Plot using Basemap’s pcolor ---
    # 1. Plot the DEM color base map
    pc = m.pcolor(lon, lat, z, latlon=True, cmap=cmap, alpha=1.0, zorder=20)
    
    # 2. Overlay the semi-transparent hillshade to add 3D relief
    m.pcolor(lon, lat, shade_gray, latlon=True, cmap='gray', alpha=0.05, zorder=21)
    
    # --- Convert catchment coordinates to map projection and plot boundary ---
    xpt, ypt = m(latlon[:, 1], latlon[:, 0])
    mlatlon = np.transpose(np.array([xpt, ypt]))
    
    # Catchments
    #facecolor = perco_color
    sub.add_patch(Polygon(mlatlon, facecolor='None', linewidth=lwidth, edgecolor='black', alpha=1.0, zorder=200))

    # Load the shapefile using geopandas
    import geopandas as gpd
    shapefile_path = 'watersheld_polygon_WGS84.shp'  # Replace with your shapefile path
    gdf = gpd.read_file(shapefile_path)
    
    # Ensure the shapefile is in WGS84 (EPSG:4326) projection
    #if gdf.crs != 'EPSG:4326':
    #    gdf = gdf.to_crs('EPSG:4326')
        
    # Overlay shapefile on the map
    # Convert Geopandas geometries to Basemap coordinates and plot them
    for geometry in gdf.geometry:
        if geometry.geom_type == 'Polygon':
            # Single Polygon: plot directly
            x, y = m(*geometry.exterior.xy)
            m.plot(x, y, marker=None, color='grey', linewidth=0.75)
        elif geometry.geom_type == 'MultiPolygon':
            # MultiPolygon: iterate through each polygon
            for polygon in geometry.geoms:  # Corrected this line to iterate over geometry.geoms
                x, y = m(*polygon.exterior.xy)
                m.plot(x, y, marker=None, color='k', linewidth=0.05)
            
    # Load the shapefile using geopandas
    import geopandas as gpd
    shapefile_path = r'D:\Jinghe_SWAT\reach_polyline_WGS84.shp'  # Replace with your shapefile path
    gdf = gpd.read_file(shapefile_path)
    
    # Convert Geopandas geometries to Basemap coordinates and plot them
    for igeo, geometry in enumerate(gdf.geometry):
        if geometry.geom_type == 'Polygon':
            # Single Polygon: plot directly
            x, y = m(*geometry.exterior.xy)
            m.plot(x, y, marker=None, color='grey', linewidth=1.0)
        elif geometry.geom_type == 'MultiPolygon':
            # MultiPolygon: iterate through each individual polygon
            for polygon in geometry.geoms:  # Corrected this line to iterate over geometry.geoms
                x, y = m(*polygon.exterior.xy)
                m.plot(x, y, marker=None, color='grey', linewidth=1.0)
        elif geometry.geom_type == 'LineString':
            # LineString (Polyline): plot directly
            x, y = m(*geometry.xy)
            #m.plot(x, y, marker=None, color='blue', 
            #           linewidth=(gdf['AreaC'].values[igeo])/gdf['AreaC'].values.max() * 2.25)
            
            for i, geometry in enumerate(gdf.geometry):

                # Coordinates
                x, y = m(*geometry.xy)
                # Current feature value
                val = gdf.loc[i, 'AreaC']   # fetch AreaC for the i-th row
                # 1. Compute bin edges (equal-intervals)
                bins = np.linspace(gdf['AreaC'].min(), gdf['AreaC'].max(), 6)  # 5 classes → 6 edges
                # 2. Digitize current value into class index (1–5)
                class_idx = np.digitize(val, bins, right=True)
                # 3. Define linewidths for 5 classes
                linewidth_classes = np.array([1.0, 1.5, 1.75, 1.875, 1.9375]) / 2.5
                # 4. Pick linewidth based on class
                lw = linewidth_classes[class_idx - 1]
                # 5. Plot
                m.plot(x, y, marker=None, color='blue', linewidth=lw, alpha=0.5)
                        

          
                                   
        elif geometry.geom_type == 'MultiLineString':
            # MultiLineString: iterate through each individual line
            for line in geometry.geoms:
                x, y = m(*line.xy)
                m.plot(x, y, marker=None, color='blue', linewidth=1.0)
                
    # some reference points
    nref = 4
    ref_latlon = np.ones([nref,2]) * -9999.0
    ref_names  = []
    ref_latlon[0,0] = 36.5833 ; ref_latlon[0,1] =  107.30  ; ref_names.append( 'HX' )   
    ref_latlon[1,0] = 35.7333 ; ref_latlon[1,1] =  107.63  ; ref_names.append( 'XF' )   
    ref_latlon[2,0] = 35.2 ; ref_latlon[2,1] =  107.8  ; ref_names.append( 'CW' ) 
    ref_latlon[3,0] = 36.0158 ; ref_latlon[3,1] =  106.2426  ; ref_names.append( 'GY' ) 

    for iref in range(nref):
            xpt1, ypt1 = m(ref_latlon[iref,1],ref_latlon[iref,0])
            sub.plot(xpt1, ypt1,
                     #linewidth=lwidth,
                     #color=color,
                     marker='^',
                     markeredgecolor='black',
                     markerfacecolor='black',
                     markersize=5.0, #msize,
                     markeredgewidth=mwidth,
                     zorder=100)
            ha = ['right', 'left']
            va = ['bottom','top']

            if iref in [0,1]:
                ha = 'right'
                dx = 0.35
            elif iref in [2]:
                ha = 'right'
                dx = -0.075
            elif iref in [3]:
                ha = 'right'
                dx = 0.45   
            x2, y2 = m(ref_latlon[iref,1]+dx,ref_latlon[iref,0])
            sub.annotate(str2tex(ref_names[iref]),
                xy=(xpt1, ypt1),  xycoords='data',
                xytext=(x2, y2), textcoords='data',
                fontsize='small',
                zorder=100,
                verticalalignment='center',horizontalalignment=ha,
                
                path_effects=[pe.withStroke(linewidth=2, foreground='white')]  # white halo for readability
                # arrowprops=dict(arrowstyle="->",relpos=(1.0,0.5),linewidth=0.4)
                )
            
    # some reference points
    nref = 1
    ref_latlon = np.ones([nref,2]) * -9999.0
    ref_names  = []
    ref_latlon[0,0] = 34.63778; ref_latlon[0,1] =  108.592152  ; ref_names.append( 'ZJS' )   

    for iref in range(nref):
            xpt1, ypt1 = m(ref_latlon[iref,1],ref_latlon[iref,0])
            sub.plot(xpt1, ypt1,
                     #linewidth=lwidth,
                     #color=color,
                     marker='s',
                     markeredgecolor='#C01C24',
                     markerfacecolor='#C01C24',
                     markersize=5.0, #msize,
                     markeredgewidth=mwidth,
                     zorder=100)
            ha = ['right', 'left']
            va = ['bottom','top']

            if iref in [0,2]:
                ha = 'right'
                dx = -0.05
            else:
                ha = 'left'
                dx = -0.1
            x2, y2 = m(ref_latlon[iref,1]+dx,ref_latlon[iref,0])
            sub.annotate(str2tex(ref_names[iref]),
                xy=(xpt1, ypt1),  xycoords='data',
                xytext=(x2, y2), textcoords='data',
                fontsize='small',
                zorder=100,
                verticalalignment='center',horizontalalignment=ha,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')]  # white halo for readability#,
                # arrowprops=dict(arrowstyle="->",relpos=(1.0,0.5),linewidth=0.4)
                )
            
    # Add a colorbar
    cb = m.colorbar(pc, location='right', pad='5%', extend='both')
    cb.set_label(str2tex('Elevation (m)'))
    
    # Fake subplot for numbering
    if doabc:
        pos = position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)
        lsub = fig.add_axes([lleft+wwidth-0.19+0.035, bbottom+hheight-0.098, 0.1, 0.1])

        lsub.set_xlim([0,1])
        lsub.set_ylim([0,1])

        # subplot numbering
        abc2plot(lsub, dxabc, dyabc, iplot, lower=True, parenthesis='both',
                     bold=True, large=True,
                     mathrm=True, usetex=usetex,
                     horizontalalignment='left', verticalalignment='top')

        lsub.set_title('')
        lsub.set_xlabel('')
        lsub.set_ylabel('')
        lsub.set_xticks([])
        lsub.set_yticks([])
        lsub.set_axis_off()
        
           
    
    

    # -------------
    # Salmon River watershed (landuse)
    # -------------
    iplot += 1
    #                           [left, bottom, width, height]
    lleft   = 0.39
    bbottom = 0.435
    wwidth  = 0.5 #0.3475
    hheight = 0.25 #0.155
    sub = fig.add_axes(np.array([ lleft, bbottom, wwidth, hheight ]))

    # Map: Canada - Lake Erie
    llcrnrlon =  105.6 #-81.25 #-85.5
    urcrnrlon =  109.5 #-77.0
    llcrnrlat =   34.2 #39.5
    urcrnrlat =   37.6
    
    
    lat_1     =   (llcrnrlat+urcrnrlat)/2.0 # 42.0  # first  "equator"
    lat_2     =   (llcrnrlat+urcrnrlat)/2.0 # 42.0  # second "equator"
    lat_0     =   (llcrnrlat+urcrnrlat)/2.0 # 42.5  # center of the map
    lon_0     =   (llcrnrlon+urcrnrlon)/2.0 #-79.00 #-82.0  # center of the map
    # m = Basemap(projection='lcc',
    #             llcrnrlon=-80, llcrnrlat=43, urcrnrlon=-75, urcrnrlat=47,
    #             lon_0=-77.5, lat_0=43, 
    #             lat_1=44, lat_2=44, 
    #             resolution='i') # Lambert conformal
    m = Basemap(projection='lcc', #area_thresh=10000.,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
                resolution='i') # Lambert conformal
    
    #m.readshapefile('Dagu_waterways','waterways', color='#2B6FAD', linewidth=1)
    
    
    # draw parallels and meridians.
    # labels: [left, right, top, bottom]
    parallels = m.drawparallels(np.arange(-80.,81., 1),  labels=[0,0,0,0], dashes=[1,1], linewidth=0.25, color='0.5', rotation=90)
    meridians = m.drawmeridians(np.arange(-180,181.1, 1),labels=[0,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
    
    # === Make latitude & longitude labels italic ===
    for par_dict in [parallels, meridians]:
        for key in par_dict:
            for text in par_dict[key][1]:  # [1] -> list of Text objects
                #text.set_fontstyle('italic')  # italic font
                text.set_fontsize(9)          # optional: set font size
                text.set_color('black')       # optional: set color
                
    for key, (line, texts) in parallels.items():
        for text in texts:
            lat = key
            if lat > 0:
                new_label = rf"${abs(lat):.0f}^\circ\mathrm{{N}}$"
            else:
                new_label = rf"${abs(lat):.0f}^\circ\mathrm{{S}}$"
            text.set_text(str2tex(new_label, usetex=True))             
            text.set_color('black')

    for key, (line, texts) in meridians.items():
        for text in texts:
            lon = key
            if lon > 0:
                new_label = rf"${abs(lon):.0f}^\circ\mathrm{{E}}$"
            else:
                new_label = rf"${abs(lon):.0f}^\circ\mathrm{{W}}$"
            text.set_text(str2tex(new_label, usetex=True))  
            text.set_color('black')
            
    
    m.etopo(scale=1.5)
    
    # draw cooastlines and countries
    # m.drawcoastlines(linewidth=0.3)
    m.drawmapboundary(fill_color=ocean_color, linewidth=0.3)
    m.drawcountries(color='black', linewidth=0.3)
    m.fillcontinents(color='white', lake_color=ocean_color)

    # scalebar
    length = 100 # km
    m.drawmapscale(llcrnrlon*0.80+urcrnrlon*0.20, llcrnrlat*0.88+urcrnrlat*0.12, lon_0, lat_0, length, barstyle='fancy', labelstyle='simple', fontsize='small')

    for text in sub.texts:
        if 'km' in text.get_text():   
            text.set_text(str2tex('km')) 
        if text.get_text() == '0':
            text.set_text(str2tex('0')) 
        if text.get_text() == '50':
            text.set_text(str2tex('50')) 
        if text.get_text() == '100':
            text.set_text(str2tex('100')) 


##############################################################################################################################


    # Load the NetCDF file
    from netCDF4 import Dataset
    file_path = 'hru_WGS84_ProjectRaster.nc'
    nc = Dataset(file_path, 'r')
    
    # Extract latitude, longitude, and z values
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    var = nc.variables['hru'][:]  # Replace 'z' with your actual variable name
    
    # Close the NetCDF file
    nc.close()
    
    # Create a meshgrid for lat and lon
    lon, lat = np.meshgrid(lon, lat)
    
    # Plot the data using m.pcolor
    cmap = plt.cm.rainbow
    pc = m.pcolor(lon, lat, var, latlon=True, cmap=cmap)
    
    # convert coordinates to map coordinates
    xpt,ypt = m(latlon[:,1], latlon[:,0])
    mlatlon = np.transpose(np.array([xpt,ypt])) # zip(xpt,ypt)

    # Catchments
    #facecolor = perco_color
    sub.add_patch(Polygon(mlatlon, facecolor='None', linewidth=lwidth, edgecolor='black', alpha=1.0, zorder=200))

    # Add a colorbar
    cb = m.colorbar(pc, location='right', pad='5%', extend='both')
    cb.set_label(str2tex('HRU ID'))
    
    # Load the shapefile using geopandas
    import geopandas as gpd
    shapefile_path = 'watersheld_polygon_WGS84.shp'  # Replace with your shapefile path
    gdf = gpd.read_file(shapefile_path)
    
    # Ensure the shapefile is in WGS84 (EPSG:4326) projection
    #if gdf.crs != 'EPSG:4326':
    #    gdf = gdf.to_crs('EPSG:4326')
        
    # Overlay shapefile on the map
    # Convert Geopandas geometries to Basemap coordinates and plot them
    for geometry in gdf.geometry:
        if geometry.geom_type == 'Polygon':
            # Single Polygon: plot directly
            x, y = m(*geometry.exterior.xy)
            m.plot(x, y, marker=None, color='grey', linewidth=0.75)
        elif geometry.geom_type == 'MultiPolygon':
            # MultiPolygon: iterate through each polygon
            for polygon in geometry.geoms:  # Corrected this line to iterate over geometry.geoms
                x, y = m(*polygon.exterior.xy)
                m.plot(x, y, marker=None, color='k', linewidth=0.05)
                
    # If your field is 'Subbasin' (capital S), switch this name accordingly
    
    field_name = 'Subbasin'  
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
    
        # --- draw polygon outlines (same as you do now) ---
        if geom.geom_type == 'Polygon':
            x, y = m(*geom.exterior.xy)
            m.plot(x, y, marker=None, color='grey', linewidth=0.5, zorder=2)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = m(*poly.exterior.xy)
                m.plot(x, y, marker=None, color='grey', linewidth=0.5, zorder=2)
    
        # --- compute a stable interior label point ---
        # representative_point() is better than centroid because it is guaranteed to lie inside
        rep_pt = geom.representative_point()
        lon, lat = rep_pt.x, rep_pt.y
        X, Y = m(lon, lat)
    
        # --- draw the text label ---
        label = str(row[field_name])
        if label in ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
                     '10', '11', '12', '13',  '14', '15', '16', 
                     '18', '20', '21', '22', '24', '26', '27', '29', '30', '31', 
                     '32', '33', '34', '35', '36', '37', '38', '39']:
            plt.text(
                X, Y, str2tex(label),
                ha='center', va='center', fontsize=7, zorder=5, clip_on=False,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')]  # white halo for readability
            )
            
    # Load the shapefile using geopandas
    import geopandas as gpd
    shapefile_path = r'D:\Jinghe_SWAT\reach_polyline_WGS84.shp'  # Replace with your shapefile path
    gdf = gpd.read_file(shapefile_path)
    
    # Convert Geopandas geometries to Basemap coordinates and plot them
    for igeo, geometry in enumerate(gdf.geometry):
        if geometry.geom_type == 'Polygon':
            # Single Polygon: plot directly
            x, y = m(*geometry.exterior.xy)
            m.plot(x, y, marker=None, color='grey', linewidth=1.0)
        elif geometry.geom_type == 'MultiPolygon':
            # MultiPolygon: iterate through each individual polygon
            for polygon in geometry.geoms:  # Corrected this line to iterate over geometry.geoms
                x, y = m(*polygon.exterior.xy)
                m.plot(x, y, marker=None, color='grey', linewidth=1.0)
        elif geometry.geom_type == 'LineString':
            # LineString (Polyline): plot directly
            x, y = m(*geometry.xy)
            #m.plot(x, y, marker=None, color='blue', 
            #           linewidth=(gdf['AreaC'].values[igeo])/gdf['AreaC'].values.max() * 1.25)
        elif geometry.geom_type == 'MultiLineString':
            # MultiLineString: iterate through each individual line
            for line in geometry.geoms:
                x, y = m(*line.xy)
                m.plot(x, y, marker=None, color='blue', linewidth=1.0)
                
     # Fake subplot for numbering
    if doabc:
        pos = position(nrow,ncol,iplot,hspace=hspace,vspace=vspace)
        lsub = fig.add_axes([lleft+wwidth-0.19+0.035, bbottom+hheight-0.098, 0.1, 0.1])

        lsub.set_xlim([0,1])
        lsub.set_ylim([0,1])

        # subplot numbering
        abc2plot(lsub, dxabc, dyabc, iplot, lower=True, parenthesis='both',
                     bold=True, large=True,
                     mathrm=True, usetex=usetex,
                     horizontalalignment='left', verticalalignment='top')

        lsub.set_title('')
        lsub.set_xlabel('')
        lsub.set_ylabel('')
        lsub.set_xticks([])
        lsub.set_yticks([])
        lsub.set_axis_off()
    
        


    outtype = 'png'
    pngbase = 'Fig1'
    if (outtype == 'pdf'):
        pdf_pages.savefig(fig)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, dpi=600)
        plt.close(fig)

    # --------------------------------------
    # Finish
    # --------------------------------------
    if (outtype == 'pdf'):
        pdf_pages.close()
    elif (outtype == 'png'):
        pass
    else:
        plt.show()

    
    t2  = time.time()
    str = '  Time plot [m]: '+astr((t2-t1)/60.,1) if (t2-t1)>60. else '  Time plot [s]: '+astr(t2-t1,0)
    print(str)


fig.savefig('Fig10001.png', dpi=600)