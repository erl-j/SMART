from symusic import BuiltInSF3

SF_PATH= {"musescore": BuiltInSF3.MuseScoreGeneral().path(download=True), 
            "sgm": "./soundfonts/SGM-V2.01-XG-2.04.sf2",
            "monalisa":"./soundfonts/Monalisa_GM_v2_105.sf2",
            "ephesus":"./soundfonts/Ephesus_GM_Version_1_00.sf2",
            "touhou" : "./soundfonts/Touhou.sf2",
            "arachno": "./soundfonts/Arachno SoundFont - Version 1.0.sf2",
            "fluidr3": "./soundfonts/FluidR3 GM.sf2",

            }[SOUNDFONT]