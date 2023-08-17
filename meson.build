project(
  'open-belex-tests', 'c',
  version: '1.0.2',
  meson_version : '>= 0.58.0'
)

if not get_option('with-baryon')
  subdir('conda_prefix/common')
  fs = import('fs')
  if not fs.is_dir('/efs/data/public/synopsis/ARC-2018.06/MetaWare/arc/inc/') \
     and fs.is_dir('/home/admin/O_2018_06/MetaWare/arc/inc')
    gvml_module_common_inc = ['/home/admin/O_2018_06/MetaWare/arc/inc']
  endif
endif

subdir('gensrc')
