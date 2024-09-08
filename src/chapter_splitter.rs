use log::info;
use mdbook::{
    book::{Book, Chapter},
    errors::Error,
    preprocess::{Preprocessor, PreprocessorContext},
    BookItem,
};
use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::{fmt, vec};
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
            let path = base_path.join(&chapter.name);
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

        vec![]
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
        Ok(book)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use mdbook::preprocess;
    use pulldown_cmark::Parser;

    use super::ChapterSplitter;
    #[test]
    fn asdrubale() {
        let preprocess = ChapterSplitter::default();
        let mut markdown_files = preprocess.get_markdown_files(PathBuf::from("./test_book/src"));
        // let markdown_files: Vec<String> = markdown_files
        //     .iter()
        //     .map(|md| md.clone().into_os_string().into_string().unwrap())
        //     .filter(|md| !md.contains("SUMMARY.md"))
        //     .collect();
        // .for_each(|md| println!("{}", md));
        for markdown in markdown_files {
            if markdown.file_name().unwrap() != "SUMMARY.md" {
                let content = fs::read_to_string(&markdown).unwrap();
                let parser = Parser::new(&content);
                for event in parser {
                    println!("{:?}", event)
                }
            }
        }
    }
}
