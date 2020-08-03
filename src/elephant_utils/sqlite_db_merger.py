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
                 tables=None,
                 verbose=False
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
            
            if verbose:
                print(f"Processing {sqlite_file}...")
            
            if tables is None:
                table_rows = in_db.execute('''
                 SELECT name
                   FROM sqlite_master
                  WHERE type = 'table' 
                ''').fetchall()
                tables = [tbl['name'] for tbl in table_rows]
            
            table_info_rows = in_db.execute('''
                SELECT tbl_name, sql 
                  FROM sqlite_master 
                WHERE type = 'table'
                ''')
            # For each table name in the current
            # input db, copy that table, if it was
            # requested via the 'tables' arg:
            for table_info_row in table_info_rows:
                tbl_name = table_info_row['tbl_name']
                if tbl_name in tables:
                    # Ensure that sample_id is kept unique
                    # in the destination via the col_to_map
                    # arg:
                    self.copy_table(table_info_row,
                                    in_db,
                                    dest_db,
                                    col_to_map='sample_id',
                                    verbose=verbose 
                                    )
            in_db.close()

            if verbose:
                print(f"Done processing {sqlite_file}...")


        dest_db.close()
            
    #------------------------------------
    # copy_table
    #-------------------

    def copy_table(self, 
                   table_info_dict,
                   src_db,
                   dst_db,
                   col_to_map=None,
                   verbose=False
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
        
        The col_to_map allows for re-numbering of integer
        primary keys to a higher range. Used to ensure that
        tables successively merged from multiple src dbs to the
        dst db remain unique in the dst. The method will find
        the highest current dest int in the specified col,
        and ensure that the next table copy begins above
        that latest val.  
        
        @param table_info_dict: 'tbl_name' and 'sql' entries
            for table name and it's sql create statement.
        @type table_info_dict: {str : str}
        @param src_db: sqlite3 connection instance to the
            source db.
        @type src_db: sqlite3.Connection
        @param dst_db: sqlite3 connection instance to the
            destination db.
        @type dst_db: sqlite3.Connection
        @param col_to_map: primary integer key column whose
            value must be mapped to a different range during
            copying
        @type col_to_map: {None|str}
        @param verbose: print debug info
        @type verbose: bool
        '''
        
        # If table to copy does not yet exist
        # at the dest, create it there:
        tbl_name = table_info_dict['tbl_name']
        if tbl_name not in self.dest_tables:
            # Create table in dest db, using 
            # the table metainfo from sqlite_master:
            dst_db.execute(table_info_dict['sql'])
            self.dest_tables.append(tbl_name)
            
        # If the src tbl has a primary int key
        # that needs to be mapped, find the last
        # value of that table in the dest:
        try:
            prim_key_offset = next(dst_db.execute(f'''
                    SELECT MAX({col_to_map}) FROM {tbl_name};
                    '''))[0]
            if prim_key_offset is None:
                # Destination tbl is empty
                prim_key_offset = 0
            # Since 0 based, map to range one higher:
            prim_key_offset += 1
        except Exception as e:
            # Either the col to be mapped isn't part
            # of the tbl being copied; fine:
            if str(e).find('no such column') > -1:
                prim_key_offset = None
        except StopIteration:
            # Or the dest table is empty:
            prim_key_offset = 0

        # Copy:
        # Get row dicts for all rows in the
        # input db. For large tables this may
        # be infeasible, and this would need
        # to be done in batches. But not in
        # our use case:
        dict_list = src_db.execute(f'''
            SELECT *
              FROM {tbl_name};
            ''').fetchall()
            
        if verbose:
            print(f"Copying {len(dict_list)} rows from table {tbl_name}")
            
        if len(dict_list) == 0:
            # Src table was empty:
            return
        
        # Src col names:
        col_names = dict_list[0].keys()
        col_name_str = ','.join(col_names)

        # Build pieces of the insert statement:
        #   INSERT INTO <tblName> 
        #      (colname1, colname2)     <--- build this
        #   VALUES (val1,val2,...)      <--- and this
        
        vals_list = []
        for info_dict in dict_list:
            # Values in one row of the src tbl:
            vals_one_row = [f"'{str(info_dict[col_name])}'" 
                            for col_name in col_names]
            # If a primary key value needs to be mapped,
            # find its position in col_names:
            if col_to_map is not None:
                try:
                    prim_key_pos = col_names.index(col_to_map)
                    prim_key_val = int(vals_one_row[prim_key_pos])
                    prim_key_val += prim_key_offset
                    vals_one_row[prim_key_pos] = str(prim_key_val)
                except ValueError:
                    # That prim key is not present in the
                    # tbl being copied:
                    pass
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
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Print progress messages; default False')
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
                       tables=args.tables,
                       verbose=args.verbose
                       )