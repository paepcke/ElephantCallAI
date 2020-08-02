#!/usr/bin/env python
'''
Created on Jul 30, 2020

@author: paepcke
'''
import os
import sqlite3
import sys

import argparse


class SqliteDbMerger(object):
    '''
    Combines tables spread across multiple
    sqlite files into a new sqlite db.
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 sqlite_infiles,
                 sqlite_outfile,
                 tables=None
                 ):
        '''
        Given a list of sqlite file names, and 
        the file name of an output sqlite file,
        copy one or more tables from all the 
        sources to the destination
        
        @param sqlite_infiles: full paths to sqlite files
        @type sqlite_infiles: [str]
        @param sqlite_outfile: full path to sqlite destination file
        @type sqlite_outfile: str
        @param tables: If None, copy all tables from all sources
            to the destination. Else copy just the named tables.
        @type tables: {None|[str]}
        '''

        # No tables created yet in dest in_db:
        self.dest_tables = []
        
        dest_db = sqlite3.connect(sqlite_outfile)
        
        for sqlite_file in sqlite_infiles:
            in_db = sqlite3.connect(sqlite_file)
            in_db.row_factory = sqlite3.Row
            
            if tables is None:
                tables = in_db.execute('''
                 SELECT name
                   FROM sqlite_master
                  WHERE type = 'table' 
                ''').fetchall()
            
            table_info_rows = in_db.execute('''
                SELECT tbl_name, sql 
                  FROM sqlite_master 
                WHERE type = 'table'
                ''')
            for table_info_row in table_info_rows:
                tbl_name = table_info_row['tbl_name']
                if tbl_name in tables:
                    self.copy_table(table_info_row,
                                    in_db,
                                    dest_db 
                                    )
            in_db.close()

        dest_db.close()
            
    #------------------------------------
    # copy_table
    #-------------------

    def copy_table(self, 
                   table_info_dict,
                   src_db,
                   dst_db
                   ):
        '''
        Copy all entries of one table into 
        a same-named table in another db.
        
        The table_info_dict is expected to 
        come from the sqlite_master of the source
        db. It must contain keys 'tbl_name' and 'sql'.
        Where 'tbl_name' is the table name, and 'sql'
        is an sql statement that when executed
        creates that table. Both quantities are straight
        out of the sqlite_master.
        
        If a table of the given table name does not
        exist in the dst_db instance, that table is
        created there, and the creation is noted in
        self.dest_tables.
        
        The entries in the from table are then inserted
        into the dest table. 
        
        @param table_info_dict: 'tbl_name' and 'sql' entries
            for table name and it's sql create statement.
        @type table_info_dict: {str : str}
        @param src_db: sqlite3 connection instance to the
            source db.
        @type src_db: sqlite3.Connection
        @param dst_db: sqlite3 connection instance to the
            destination db.
        @type dst_db: sqlite3.Connection
        '''
        
        tbl_name = table_info_dict['tbl_name']
        if tbl_name not in self.dest_tables:
            # Create table in dest db:
            dst_db.execute(table_info_dict['sql'])
            self.dest_tables.append(tbl_name)
            
        # Copy:
        # Get row dicts for all rows:
        dict_list = src_db.execute(f'''
            SELECT *
              FROM {tbl_name};
            ''').fetchall()
        if len(dict_list) == 0:
            return
        
        col_names = dict_list[0].keys()
        col_name_str = ','.join(col_names)
        
        vals_list = []
        for info_dict in dict_list:
            vals_one_row = [f"'{str(info_dict[col_name])}'" 
                            for col_name in col_names]
            vals_list.append(f"({','.join(vals_one_row)})")
             
        col_vals_str = ','.join(vals_list)
        insert_cmd = f'''INSERT INTO {tbl_name}
                    ({col_name_str})
                    VALUES {col_vals_str};'''
        dst_db.execute(insert_cmd)
        dst_db.commit()
        
# ------------------------ Main ------------
if __name__ == '__main__':
    
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         description="Merges multiple sqlite dbs into one"
                                         )
    
        parser.add_argument('-t', '--tables',
                            type=str,
                            nargs='+',
                            help='Repeatable: tables to copy from src dbs to dst db; default: all tables')
        parser.add_argument('dbfiles',
                            type=str,
                            nargs='+',
                            help='Repeatable: paths to src sqlite files; last is destination sqlite file')

    
        args = parser.parse_args();

        if len(args.dbfiles) < 2:
            print("At least two sqlite files must be provided: one src db and the destination db.")
            sys.exit(1)
            
        SqliteDbMerger(args.dbfiles[:-1], # infiles
                       args.dbfiles[-1],  # outfile
                       tables=args.tables
                       )