use log::info;
use mdbook::{
    book::{Book, Chapter},
    errors::Error,
    preprocess::{Preprocessor, PreprocessorContext},
    BookItem,
};
use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::{fs::OpenOptions, path::Display};
use walkdir::{DirEntry, WalkDir};

pub struct ChapterSplitter;
impl ChapterSplitter {
    pub fn new() -> ChapterSplitter {
        ChapterSplitter
    }
    fn log_to_file(&self, message: &str) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("preprocessor.log")
            .expect("Unable to open log file");

        writeln!(file, "{}", message).expect("Unable to write to log file");
    }
    fn create_structure(&self, chapters: &[Chapter], base_path: &Path) {
        for chapter in chapters {
            let path = base_path.join(&chapter.title);
            fs::create_dir_all(&path).expect("Failed to create directory");
            let file_path = path.join("introduction.md");
            fs::write(file_path, &chapter.content).expect("Failed to write file");
        }
    }
    fn get_markdown_files<T: AsRef<Path>>(&self, src_dir: T) -> Vec<PathBuf> {
        WalkDir::new(src_dir)
            .into_iter()
            .filter_map(Result::ok) // Ignore any errors during directory traversal
            .filter(|entry| {
                entry.file_type().is_file()
                    && entry.path().extension().map_or(false, |ext| ext == "md")
            }) // Filter for Markdown files
            .map(|entry| entry.into_path())
            .collect()
    }
    fn parse_markdown(&self, content: &str) -> Vec<Chapter> {
        let parser = Parser::new(content);
        // let mut chapters = vec![];
        // let mut current_chapter = Chapter::default();
        vec![]
        // for event in parser {
        //     match event {
        //         Event::Start(Tag::Heading { level, .. }) => {
        //             // Handle new chapter or section
        //             let current_level = level as u8;
        //             if !current_chapter.name.is_empty() {
        //                 chapters.push(current_chapter);
        //                 current_chapter = Chapter::default();
        //             }
        //         }
        //         Event::Text(text) => {
        //             // Collect text for the current heading
        //             if current_chapter.title.is_empty() {
        //                 current_chapter.title = text.to_string();
        //             } else {
        //                 current_chapter.content.push_str(&text);
        //             }
        //         }
        //         // Event::End(TagEnd::Heading(_)) => {
        //         //     // Finalize the current section
        //         //     current_chapter.content.push('\n');
        //         // }
        //         _ => {}
        //     }
        // }
        // if !current_chapter.name.is_empty() {
        //     chapters.push(current_chapter);
        // }
        // chapters
    }
}
impl Default for ChapterSplitter {
    fn default() -> Self {
        ChapterSplitter
    }
}
impl Preprocessor for ChapterSplitter {
    fn name(&self) -> &str {
        "chapter_splitter"
    }

    fn run(&self, ctx: &PreprocessorContext, mut book: Book) -> Result<Book, Error> {
        let src_dir = ctx.root.join(&ctx.config.book.src);
        let markdown_files = self.get_markdown_files(src_dir);

        for markdown_path in markdown_files {
            // self.log_to_file(
            //     &markdown_path
            //         .clone()
            //         .into_os_string()
            //         .into_string()
            //         .unwrap(),
            // );
            // self.log_to_file(&"\n\n");
            let chapters = self.parse_markdown(&fs::read_to_string(markdown_path).unwrap());
            for chapter in chapters {
                self.log_to_file(&chapter.title);
                self.log_to_file(&chapter.content);
                self.log_to_file(&"\n\n");
            }
        }
        // markdown_files
        //     .iter()
        //     .for_each(|md| log_to_file(&md.clone().into_os_string().into_string().unwrap()));

        Ok(book)
    }
}
